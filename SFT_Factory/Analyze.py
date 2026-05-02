import json
import os
import glob
import hashlib
import random
import time
from openai import OpenAI

# ================= 请根据你的实际情况修改以下配置 =================
DEEPSEEK_API_KEY = "sk-979570db57044e5c95fd2c3bbe2def59"          # 你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"   # DeepSeek API 地址
MODEL_NAME = "deepseek-v4-pro"                   # 使用的模型名称
FOLDER_PATH = ".."  # 存放 JSON 文件的文件夹路径
OUTPUT_FILE = "analyze_comprehension.json"    # 输出文件名（保存在同一文件夹下）
SLEEP_INTERVAL = 0.5                             # API 调用间隔（秒）
RANDOM_SEED = 42                                 # 随机种子（固定可复现，若想完全随机可设为 None）
MAX_POEMS = 1000                                 # 最多处理 1000 首（不足则全部）
# ================================================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

SYSTEM_PROMPT = (
    "你是中国古典诗词赏析专家，能依据诗词内容与语境，对给定选项进行辨析，"
    "判断其中表述最为合理的一项。回答时请先给出推理过程，再给出最终答案。"
)

# -------------------------------------------------------------------
# 1. 扫描文件夹，找出所有文件名包含 "ci" 或 "poet" 的 JSON 文件
# -------------------------------------------------------------------
def find_json_files(folder):
    pattern_ci = os.path.join(folder, "*ci*.json")
    pattern_poet = os.path.join(folder, "*poet*.json")
    files = glob.glob(pattern_ci) + glob.glob(pattern_poet)
    return list(set(files))   # 去重

# -------------------------------------------------------------------
# 2. 从单个 JSON 文件中递归提取所有包含 "paragraphs" 键的诗歌对象
# -------------------------------------------------------------------
def extract_poems_from_file(file_path):
    poems = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 失败：{e}")
        return poems

    def traverse(obj):
        if isinstance(obj, dict):
            if "paragraphs" in obj and obj["paragraphs"]:
                title = obj.get("title", "untitled")
                author = obj.get("author", "unknown")
                paragraphs = obj["paragraphs"]
                if isinstance(paragraphs, list):
                    text = "\n".join(str(p) for p in paragraphs if p)
                elif isinstance(paragraphs, str):
                    text = paragraphs
                else:
                    text = ""
                if text.strip():
                    # 生成唯一 ID：文件路径 + 标题 + 作者 + 内容前200字符的哈希
                    content_hash = hashlib.md5(
                        (file_path + title + author + text[:200]).encode("utf-8")
                    ).hexdigest()
                    poem_id = f"{os.path.basename(file_path)}::{title}::{author}::{content_hash[:8]}"
                    poems.append({
                        "id": poem_id,
                        "file": file_path,
                        "title": title,
                        "author": author,
                        "paragraphs_text": text
                    })
            # 递归处理字典的值
            for value in obj.values():
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)

    traverse(data)
    # 为同一文件内的诗歌补充序号（仅用于展示）
    for i, p in enumerate(poems):
        p["index"] = i
    return poems

# -------------------------------------------------------------------
# 3. 遍历所有符合条件的文件，提取全部诗歌
# -------------------------------------------------------------------
def load_all_poems(folder):
    files = find_json_files(folder)
    print(f"找到 {len(files)} 个符合条件的 JSON 文件")
    all_poems = []
    for file_path in files:
        poems = extract_poems_from_file(file_path)
        print(f"  文件 {os.path.basename(file_path)} 中提取到 {len(poems)} 首诗歌")
        all_poems.extend(poems)
    print(f"总共提取到 {len(all_poems)} 首诗歌")
    return all_poems

# -------------------------------------------------------------------
# 4. 断点续传：读取已处理的诗歌 ID
# -------------------------------------------------------------------
def get_processed_ids(output_file):
    if not os.path.exists(output_file):
        return set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set()
        for entry in data:
            if "_meta" in entry and "poem_id" in entry["_meta"]:
                processed.add(entry["_meta"]["poem_id"])
        return processed
    except Exception as e:
        print(f"读取 {output_file} 时出错：{e}，将从头开始生成。")
        return set()

# -------------------------------------------------------------------
# 5. 调用 DeepSeek API 生成选择题
# -------------------------------------------------------------------
def generate_question_and_answer(poem_text, title, author):
    user_prompt = f"""请根据下面这首诗歌，生成一道选择题。要求如下：

1. 题面固定为：“下列对这首诗的理解和赏析，不正确的一项是（  ）”
2. 提供四个选项 A、B、C、D，其中只有一个选项的理解或赏析是错误的（另外三个正确）。
3. 四个选项应围绕诗歌内容、意境、手法、情感、典故等方面设置，难度适中。
4. 在你生成答案后，请先给出详细的推理分析过程，说明为什么某个选项不正确，最后给出最终答案（例如“答案：A”）。

输出格式必须严格遵守以下 JSON 格式，不要包含任何额外解释：
{{
    "options": ["A. 第一个选项内容", "B. 第二个选项内容", "C. 第三个选项内容", "D. 第四个选项内容"],
    "analysis": "这里填入你的推理分析过程（流畅的现代汉语）",
    "answer": "A"
}}

诗歌信息：
标题：{title}
作者：{author}
正文：
{poem_text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        for key in ["options", "analysis", "answer"]:
            if key not in result:
                raise ValueError(f"缺少字段: {key}")
        return result
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None

# -------------------------------------------------------------------
# 6. 保存结果（支持断点续传，记录 _meta 信息）
# -------------------------------------------------------------------
def save_result(output_file, poem_id, title, author, poem_text, qa_result):
    # 严格按照要求拼接 user 内容
    options_str = "\n".join(qa_result["options"])
    user_content = f"请分析这首诗：“{poem_text}”，并告诉我下列选项中不正确的一项：{options_str}"
    assistant_content = f"推理过程：{qa_result['analysis']}\n最终答案：{qa_result['answer']}"

    new_entry = {
        "_meta": {
            "poem_id": poem_id,
            "title": title,
            "author": author
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

    # 读取现有数据
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except:
            existing = []
    else:
        existing = []

    existing.append(new_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"已保存诗歌《{title}》的生成结果到 {output_file}")

# -------------------------------------------------------------------
# 7. 主流程
# -------------------------------------------------------------------
def main():
    # 提取全部诗歌
    all_poems = load_all_poems(FOLDER_PATH)
    if not all_poems:
        print("未找到任何诗歌，请检查文件夹路径和文件内容。")
        return

    # 从全部诗歌中随机选取最多 MAX_POEMS 首
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    sample_size = min(MAX_POEMS, len(all_poems))
    selected_poems = random.sample(all_poems, sample_size)
    print(f"从总计 {len(all_poems)} 首诗歌中随机选取了 {len(selected_poems)} 首进行处理")

    # 断点续传：跳过已处理的
    processed_ids = get_processed_ids(OUTPUT_FILE)
    remaining_poems = [p for p in selected_poems if p["id"] not in processed_ids]
    print(f"之前已完成 {len(processed_ids)} 首，还有 {len(remaining_poems)} 首待处理")

    # 逐首调用 API
    for idx, poem in enumerate(remaining_poems, 1):
        print(f"\n处理进度：{idx}/{len(remaining_poems)} - 《{poem['title']}》 （{poem['author']}）")
        qa_result = generate_question_and_answer(
            poem["paragraphs_text"],
            poem["title"],
            poem["author"]
        )
        if qa_result is None:
            print(f"生成失败，跳过该诗")
            continue

        save_result(
            OUTPUT_FILE,
            poem["id"],
            poem["title"],
            poem["author"],
            poem["paragraphs_text"],
            qa_result
        )
        time.sleep(SLEEP_INTERVAL)

    print("\n全部处理完成！")

if __name__ == "__main__":
    main()