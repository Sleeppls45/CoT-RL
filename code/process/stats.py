import json

import numpy as np
from transformers import AutoTokenizer


def stats(file_path, model_name):
    total_lines = 0
    token_counts = []  # 存储每行的 token 数量
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data.get("conversations", [])

            content = None
            for conv in conversations:
                if conv.get("role") == "assistant" and "content" in conv:
                    content = conv["content"]
                    break

            # 对格式化后的文本进行分词并统计 token 数量
            tokens = tokenizer.tokenize(content)
            token_count = len(tokens)

            # 更新统计信息
            total_lines += 1
            token_counts.append(token_count)

    # 计算平均 token 数
    avg_tokens = sum(token_counts) / total_lines

    # 统计输出 token 占比
    bins = [0, 2048, 4096, 6144, 8192, 16384]  # 定义档位
    counts, edges = np.histogram(token_counts, bins=bins)
    percentages = counts / total_lines * 100  # 计算占比
    bin_labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]  # 档位标签

    # 返回统计结果
    return {
        "total_lines": total_lines,
        "avg_tokens": avg_tokens,
        "bins": bin_labels,
        "counts": counts.tolist(),
        "percentages": percentages.tolist()
    }

file_path = "../../datasets/cot/code2text/short.jsonl"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
res = stats(file_path, model_name)

print(f"Total lines: {res['total_lines']}")
print(f"Average tokens: {res['avg_tokens']:.2f}")

for label, count, percentage in zip(res["bins"], res["counts"], res["percentages"]):
    print(f"{label}: {percentage:.2f}%({count})")
