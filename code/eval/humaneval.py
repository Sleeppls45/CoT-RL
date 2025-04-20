import multiprocessing
import os
import re
import signal
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return text
    matches = [match.strip() for match in matches]
    return max(matches, key=len)


# 代码生成函数
def generate_code(prompt, do_sample=True, num_samples=1):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=do_sample,
        num_return_sequences=num_samples
    )

    texts = []
    for output in outputs:
        text = tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)
        texts.append(extract(text))

    return texts


# 检查函数
def check_correctness(problem, completion, timeout=3.0):
    def unsafe_execute():
        temp_dir = tempfile.TemporaryDirectory()
        original_dir = os.getcwd()
        os.chdir(temp_dir.name)

        check_program = (
            problem["prompt"] + "\n" +
            completion + "\n" +
            problem["test"] + "\n" +
            f"check({problem['entry_point']})"
        )

        try:
            exec_globals = {}
            signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("Timed out!")))
            signal.alarm(int(timeout))
            exec(check_program, exec_globals)
            signal.alarm(0)
            result.append("passed")
        except TimeoutError:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")
        finally:
            os.chdir(original_dir)
            temp_dir.cleanup()

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0]
    )


# 评估函数
def evaluate_pass_at_k(dataset, k_values=[1, 10]):
    total_tasks = len(dataset)
    results = {"greedy": 0}
    for k in k_values:
        results[f"pass@{k}"] = 0

    for case in dataset:
        # Greedy 解码
        sample = generate_code(case["prompt"], do_sample=False)
        correctness = check_correctness(case, sample[0])
        if correctness["passed"]:
            results["greedy"] += 1

        # Pass@k 评估
        passed = {k: False for k in k_values}
        for k in k_values:
            samples = generate_code(case["prompt"], num_samples=k)
            for sample in samples:
                if not passed[k]:
                    correctness = check_correctness(case, sample)
                    if correctness["passed"]:
                        passed[k] = True
                        results[f"pass@{k}"] += 1
                        break

    metrics = {key: value / total_tasks for key, value in results.items()}
    return metrics


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda")

dataset = load_dataset("parquet", data_files="../../datasets/humaneval/test.parquet", split="train")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)
model.eval()

metrics = evaluate_pass_at_k(dataset)
for key, value in metrics.items():
    print(f"{key}: {value:.2%}")
