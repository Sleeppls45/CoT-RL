from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


def data_formatter(example):
    # 使用模型自带的chat模板格式化对话
    formatted = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted}


def data_collator(examples):
    batch = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    batch["labels"] = batch["input_ids"].clone()
    return batch


dataset = load_dataset("json", data_files="../../datasets/cot/code2text/short_2048.jsonl", split="train")
dataset = dataset.map(data_formatter, remove_columns=["id", "conversations"])
train_data, eval_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    # train
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=80,
    weight_decay=0.01,
    # log
    report_to="tensorboard",
    logging_dir="../../log/sft",
    logging_steps=20,
    # save
    save_total_limit=5,
    output_dir="../../checkpoints/sft/code2text/short_2048",
    save_strategy="steps",
    save_steps=100,
    # eval
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=100,
    # else
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)
trainer.train()
