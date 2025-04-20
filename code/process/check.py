import json

def count_lines(path):
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        for _ in file:
            count += 1
    return count


def check_content_format(file_path):
    """
    检查文件中每行的 assistant/content 是否符合以下格式：
    - 以 '<think>\n' 开头；
    - 中间包含一段非空内容；
    - 然后是 '</think>\n\n'；
    - 最后是一段非空内容。
    输出不符合格式的行号及其 content 的最后 30 个字符。
    """
    invalid_lines = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            data = json.loads(line.strip())
            # 遍历 conversations 列表，查找符合条件的内容
            for conversation in data.get("conversations", []):
                if conversation.get("role") == "assistant":
                    content = conversation.get("content", "")
                    # 检查是否符合格式
                    if not (
                        content.startswith("<think>\n") and
                        "</think>\n\n" in content and
                        len(content.split("</think>\n\n")) == 2 and
                        content.split("<think>\n")[1].split("</think>\n\n")[0].strip() and
                        content.split("</think>\n\n")[1].strip()
                    ):
                        # 记录行号和最后 30 个字符
                        last_30_chars = content[-30:] if len(content) >= 30 else content
                        invalid_lines.append((line_number, last_30_chars))
    return invalid_lines

# 定义文件路径
file_path = "../../datasets/cot/code2text/long.jsonl"
# 统计文件行数
line_count = count_lines(file_path)
# 检查 content 格式
invalid_lines = check_content_format(file_path)

# 输出结果
print(f"The file '{file_path}' has {line_count} lines.")
if invalid_lines:
    print(f"Invalid format in file ({file_path}):")
    for line_number, last_30_chars in invalid_lines:
        print(f"  Line {line_number}: Last 30 chars -> {repr(last_30_chars)}")
else:
    print(f"All lines in the file have valid format.")
