import json

from transformers import AutoTokenizer

# 加载分词器
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 输入和输出文件路径
input_file = "../../datasets/cot/code2text/short.jsonl"
output_file = "../../datasets/cot/code2text/short_2048.jsonl"

# 最大 token 数限制
max_token_limit = 2048

# 打开输入文件并逐行读取
filtered_count = 0  # 统计筛选后的记录数
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析 JSONL 中的每一行
        data = json.loads(line.strip())
        conversations = data.get("conversations", [])

        # 使用 apply_chat_template 将 conversations 转换为 text
        formatted_text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        # 对格式化后的文本进行分词并统计 token 数量
        tokens = tokenizer.tokenize(formatted_text)
        token_count = len(tokens)

        # 如果 token 数小于等于限制，则写入输出文件
        if token_count <= max_token_limit:
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            filtered_count += 1  # 更新筛选后的记录数

# 输出新文件的行数
print(f"筛选完成，结果已保存到 {output_file}")
print(f"新文件的行数：{filtered_count}")
