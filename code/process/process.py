import json
import sys
from datetime import datetime

from datasets import load_dataset
from openai import OpenAI


def process(example):
    # 构造prompt
    prompt = f"Please convert the following code into a method description.\n\n{example['code']}"
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.0,
            stream=True
        )
        reasoning = ""
        content = ""
        for chunk in response:
            # 检查是否有内容生成
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                reasoning_chunk = getattr(delta, "reasoning_content", "") or ""
                content_chunk = getattr(delta, "content", "") or ""
                # 流式输出 reasoning
                if reasoning_chunk:
                    print(reasoning_chunk, end=" ", flush=True)
                    reasoning += reasoning_chunk
                # 流式输出 content
                if content_chunk:
                    print(content_chunk, end=" ", flush=True)
                    content += content_chunk
        # 构造保存的结果
        res = {
            "id": example["id"],
            "conversations": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning}</think>\n\n{content}"
                }
            ]
        }
        with open("../../datasets/cot/code2text/long.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"\nProcessed and saved data {example['id']}.\n")
    except TimeoutError as e:
        print(f"Timeout occurred: {e}. Exiting with status 1.")
        sys.exit(1)  # 超时退出
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Exiting with status 1.")
        sys.exit(1)  # 异常退出


client = OpenAI(api_key="sk-83eea2cb3948406bbb7b1c6a6d984be6", base_url="https://api.deepseek.com")
dataset = load_dataset("json", data_files="../../datasets/codexglue/train.jsonl", split="train")
with open("../../datasets/cot/code2text/long.jsonl", "r", encoding='utf-8') as f:
    existing_lines = f.readlines()
    start_id = len(existing_lines)
if start_id < 5000:
    dataset = dataset.select(range(start_id, 5000))
    for data in dataset:
        if datetime.now().hour > 7:
            sys.exit(0)
        process(data)
