import os
import re

import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(prompt):
    input = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    output = model.generate(
        **input,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0][len(input.input_ids[0]):], skip_special_tokens=True)
    return text


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda")

dataset = load_dataset("parquet", data_files="../../datasets/codexglue/test.parquet", split="train")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)
model.eval()

predictions = []
references = []
for item in tqdm(dataset, desc="Processing", unit="item"):
    code = re.sub(r'"""(?:\\.|[^"\\])*?"""', '', item['code'], flags=re.DOTALL)
    prompt = f"Please convert the following code into a method description.\n\n{code}"
    text = generate(prompt)
    predictions.append(text)
    references.append([item['docstring']])

# 使用 sacreBLEU 计算分数
bleu = BLEU()
score = bleu.corpus_score(predictions, references)
print(f"BLEU Score: {score.score:.2f}")
