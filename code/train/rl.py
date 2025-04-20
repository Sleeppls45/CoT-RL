import os
import re
import signal
from contextlib import contextmanager

import np
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


@contextmanager
def timeout(time=3.0):
    # 设置信号处理函数
    signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(Exception("Timed out!")))
    signal.alarm(int(time))
    try:
        yield
    finally:
        # 恢复信号处理函数为默认行为
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def extract(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return text
    matches = [match.strip() for match in matches]
    return max(matches, key=len)


def get_rewards(problem, responses):
    test_setup_codes = problem["test_setup_code"]
    test_lists = problem["test_list"]
    responses = [extract(tokenizer.decode(r, skip_special_tokens=True)) for r in responses]

    rewards = []
    for i in range(len(responses)):
        local_namespace = {}
        code = test_setup_codes[i] + "\n" + responses[i]
        try:
            with timeout():
                compile(code, "<string>", "exec")
                exec(code, globals(), local_namespace)

            test_list = test_lists[i]
            passed_tests = 0
            for case in test_list:
                try:
                    with timeout():
                        exec(case, globals(), local_namespace)
                    passed_tests += 1
                except Exception:
                    pass

            # 计算奖励分数，保留三位小数
            score = round(5 * passed_tests / len(test_list), 3)
            rewards.append(torch.tensor(score))

        except Exception:
            rewards.append(torch.tensor(-1.0))

    return rewards


# 1. Load a pretrained model
device = torch.device("cuda")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
checkpoint_path = "../../checkpoints/sft/text2code/short_2048"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, torch_dtype="auto")
# 包装模型以支持 PPO 训练，并确保在 GPU 上
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model).to(device)
policy_ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model).to(device)

# 2. Initialize trainer
config = PPOConfig(
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5
)
ppo_trainer = PPOTrainer(config, policy_model, policy_ref_model, tokenizer)

# 3. Initialize dataset
log_dir = "../../log/rl"
output_dir = "../../checkpoints/rl"
dataset = load_dataset("parquet", data_files="../../datasets/mbpp/train.parquet", split="train")
dataset = dataset.select(range(368))

# 4. 开始训练循环
num_epochs = 3
global_step = 0
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx in range(0, len(dataset), config.batch_size):
        print(f"Batch {batch_idx // config.batch_size + 1}/{len(dataset) // config.batch_size}")
        batch_data = dataset[batch_idx:batch_idx + config.batch_size]

        prompts = []
        for prompt, test_list in zip(batch_data["text"], batch_data["test_list"]):
            test_list = '\n'.join(test_list)
            prompt = (
                f"You are an expert Python programmer, and here is your task: {prompt}\n"
                f"Your code should pass these tests:\n{test_list}"
            )
            prompts.append(prompt)

        querys = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
        querys = [querys[i].to(device) for i in range(len(querys))]

        responses = ppo_trainer.generate(
            querys,
            return_prompt=False,
            max_new_tokens=1024,
            do_sample=False
        )
        responses = [r.to(device) for r in responses]

        rewards = get_rewards(batch_data, responses)
        rewards = [r.to(device) for r in rewards]
        print(f"Rewards: {rewards}")

        train_stats = ppo_trainer.step(querys, responses, rewards)

        for key, value in train_stats.items():
            try:
                # 跳过不需要记录的字段
                if "time/" in key:  # 跳过时间相关的字段（可选）
                    continue

                # 处理标量值
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Training/{key}", value, global_step)

                # 处理张量或数组
                elif isinstance(value, torch.Tensor):
                    scalar_value = value.mean().item()  # 取平均值并转换为标量
                    writer.add_scalar(f"Training/{key}", scalar_value, global_step)

                elif isinstance(value, (list, tuple, np.ndarray)):
                    scalar_value = sum(value) / len(value)  # 计算平均值
                    writer.add_scalar(f"Training/{key}", scalar_value, global_step)

                else:
                    print(f"Skipping unsupported type for key '{key}': {type(value)}")

            except Exception as e:
                print(f"Error processing key '{key}' with value '{value}': {e}")

        # 更新全局步数
        global_step += 1
    ppo_trainer.save_pretrained(os.path.join(output_dir, f"epoch_{epoch}"))

# 关闭 TensorBoard writer
writer.close()
