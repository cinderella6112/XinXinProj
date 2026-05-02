import os
import json
import glob
import random
import hashlib
import time
from openai import OpenAI

# ==================== 用户配置区域 ====================
# 1. DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-0c3f93263afd4877a422690219514805"          # 替换为您的 API Key
DEEPSEEK_MODEL = "deepseek-v4-pro"                    # 使用的模型名称

# 2. 数据文件夹路径（存放包含 "ci" 或 "poet" 的 json 文件）
DATA_FOLDER =r"D:\XinXinProj\data"             # 替换为实际文件夹路径

# 3. 输出文件路径
OUTPUT_FILE = "../SFT_DataSet/apprehend_comprehension.json"  # 输出文件名（与脚本同文件夹）

# 4. 随机选取的诗歌数量（不足则取全部）
TARGET_COUNT = 500

# 5. 随机种子（保证可复现，断点重连时诗歌列表一致）
RANDOM_SEED =64

# 6. 保存已选中诗歌列表的文件（用于断点重连）
SELECTED_POEMS_FILE = "../selected_poems_list.json"

# 7. API 调用重试设置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
SLEEP_TIME = 0.5
# ===================================================

# 初始化 OpenAI 客户端（指向 DeepSeek）
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def load_poems_from_folder(folder_path):
    """
    读取文件夹下所有文件名包含 'ci' 或 'poet' 的 json 文件，
    提取每个文件中的 'paragraphs' 字段内容作为一首诗。
    支持两种 JSON 结构：
      - 顶层为对象，直接包含 'paragraphs' 键
      - 顶层为数组，每个元素包含 'paragraphs' 键
    返回诗歌列表，每个元素为 (来源文件, 诗歌原文)
    """
    poems = []
    pattern = os.path.join(folder_path, "*.json")
    for file_path in glob.glob(pattern):
        file_name = os.path.basename(file_path).lower()
        if "ci" not in file_name and "poet" not in file_name:
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 情况1：data 是字典，且包含 paragraphs 键
            if isinstance(data, dict) and "paragraphs" in data:
                paragraphs = data["paragraphs"]
                if isinstance(paragraphs, list):
                    poem_text = "\n".join(paragraphs)
                elif isinstance(paragraphs, str):
                    poem_text = paragraphs
                else:
                    continue
                if poem_text.strip():
                    poems.append((file_path, poem_text))
            
            # 情况2：data 是列表，遍历每个元素提取 paragraphs
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict) and "paragraphs" in item:
                        paragraphs = item["paragraphs"]
                        if isinstance(paragraphs, list):
                            poem_text = "\n".join(paragraphs)
                        elif isinstance(paragraphs, str):
                            poem_text = paragraphs
                        else:
                            continue
                        if poem_text.strip():
                            # 用文件路径加索引区分同一文件中的多首诗
                            unique_id = f"{file_path}::{idx}"
                            poems.append((unique_id, poem_text))
            else:
                print(f"警告: 文件 {file_path} 的结构无法识别（非对象也非数组），跳过")
                continue
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")
    return poems

def select_or_load_poems(poems, target_count, seed, selected_file):
    """
    随机选取 target_count 首诗歌，并将选中的诗歌列表保存到 selected_file。
    如果 selected_file 已存在，则直接加载，保证断点重连时列表一致。
    返回选中的诗歌列表（每个元素为 (来源文件, 诗歌原文)）
    """
    if os.path.exists(selected_file):
        with open(selected_file, "r", encoding="utf-8") as f:
            selected = json.load(f)
        print(f"从 {selected_file} 加载已选中的 {len(selected)} 首诗歌")
        # 将列表中的元组还原（存储时是 [["file1","text1"], ...]）
        selected = [(item[0], item[1]) for item in selected]
        return selected
    else:
        random.seed(seed)
        if len(poems) <= target_count:
            selected = poems
        else:
            selected = random.sample(poems, target_count)
        # 保存为可序列化的格式
        to_save = [[file_path, poem_text] for file_path, poem_text in selected]
        with open(selected_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"已随机选取 {len(selected)} 首诗歌，并保存至 {selected_file}")
        return selected

def load_existing_output(output_file):
    """
    加载已有的输出文件，返回已处理的消息列表（每个元素为一个完整的 messages 记录）。
    若文件不存在或为空，返回空列表。
    """
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return []
    except:
        return []

def save_output_entry(output_file, existing_data, new_entry):
    """
    将一条新记录追加到现有数据中，并写回文件。
    """
    existing_data.append(new_entry)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

def call_deepseek_api(prompt, system_content):
    """
    调用 DeepSeek API，支持重试。
    返回 API 返回的 content 字符串。
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

def generate_question_and_answer(poem_text):
    """
    针对一首诗，调用 DeepSeek API 生成一道理解性默写填空题，
    以及推理过程和答案。
    返回 (question, reasoning, answer) 三元组。
    """
    system_content = "你是中国古典诗词赏析专家，能发现古诗词中不同事物之间的相同关系，意象的关联。回答时请先给出推理过程，再给出最终答案。"
    
    user_prompt = f"""请根据以下古典诗词，完成理解性默写填空题的创作。

诗歌原文：
{poem_text}

任务要求：
1. 从这首诗中选出两句（最好是连续的两句），这两句诗中包含某种意象关联或事物之间的相同关系（比如用一事物比喻另一事物，或者两样事物并列表达某种情感）。
2. 设计一道理解性默写填空题。题目应当描述一种特定的意境、情感或画面，使读者能联想到那两句诗。题目格式示例：“《xxx》中，通过描写……的景象，表达……的两句是______。”
3. 给出解答这道题的分析过程，说明为什么填那两句诗（重点阐释其中意象的关联或相同关系）。
4. 给出最终答案（即那两句诗的原句，包含标点符号）。

请严格以 JSON 格式输出，包含三个字段：
- "question": 题目文本
- "reasoning": 分析过程（不低于50字）
- "answer": 诗句原文（完整两句）

输出示例：
{{
    "question": "《春江花月夜》中，诗人运用月与江水的交融，表现宇宙永恒、人生短暂的哲思的两句是______。",
    "reasoning": "诗中‘江畔何人初见月？江月何年初照人？’通过月与江水的意象，将自然永恒与人生短暂进行对比……",
    "answer": "江畔何人初见月？江月何年初照人？"
}}

请直接输出 JSON，不要添加额外解释。
"""
    
    response = call_deepseek_api(user_prompt, system_content)
    # 解析 JSON
    try:
        # 提取 JSON 部分（可能包含前后文字）
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("未找到有效的 JSON")
        json_str = response[start:end]
        result = json.loads(json_str)
        question = result.get("question", "")
        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")
        return question, reasoning, answer
    except Exception as e:
        print(f"解析 API 响应失败: {e}\n原始响应: {response}")
        # 返回一个降级结果
        return ("生成题目失败", "解析出错", "请检查API返回")

def build_messages_entry(question, reasoning, answer):
    """
    按照要求的格式构建一个 messages 条目。
    system 内容固定，user 为题目，assistant 为推理过程 + 最终答案。
    """
    system_content = "你是中国古典诗词赏析专家，能发现古诗词中不同事物之间的相同关系，意象的关联。回答时请先给出推理过程，再给出最终答案。"
    assistant_content = f"推理过程：{reasoning}\n\n最终答案：{answer}"
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def main():
    # 1. 读取所有诗歌
    print("正在读取诗歌数据...")
    all_poems = load_poems_from_folder(DATA_FOLDER)
    print(f"共找到 {len(all_poems)} 首含有 'paragraphs' 的诗歌")
    if not all_poems:
        print("未找到任何有效诗歌，请检查文件夹路径和文件内容。")
        return
    
    # 2. 随机选取指定数量（或全部），并保存/加载已选列表
    selected_poems = select_or_load_poems(all_poems, TARGET_COUNT, RANDOM_SEED, SELECTED_POEMS_FILE)
    print(f"待处理诗歌数量: {len(selected_poems)}")
    
    # 3. 加载已有的输出，获取已处理数量
    existing_entries = load_existing_output(OUTPUT_FILE)
    processed_count = len(existing_entries)
    print(f"已有 {processed_count} 条记录，将从第 {processed_count+1} 首开始继续处理")
    
    # 4. 遍历未处理的诗歌
    for idx in range(processed_count, len(selected_poems)):
        file_path, poem_text = selected_poems[idx]
        print(f"正在处理第 {idx+1}/{len(selected_poems)} 首: {os.path.basename(file_path)}")
        
        try:
            # 调用 API 生成题目和答案
            question, reasoning, answer = generate_question_and_answer(poem_text)
            if not question or not answer:
                print(f"  警告: 生成结果不完整，跳过该诗")
                continue
            
            # 构建符合格式的条目
            entry = build_messages_entry(question, reasoning, answer)
            
            # 动态存储
            save_output_entry(OUTPUT_FILE, existing_entries, entry)
            # 注意：existing_entries 已被 save_output_entry 内部更新（因为传入的是列表引用）
            # 但我们重新加载一下以确保一致性（简单起见可以不用，因为已经直接操作了原列表）
            # 实际 save_output_entry 中 existing_data.append(new_entry) 会修改原列表，所以不用额外操作
            
            print(f"  成功生成并保存，当前总记录数: {len(existing_entries)}")
            # 稍作延时，避免 API 限流
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"  处理诗歌时出错: {e}，跳过继续")
            continue
    
    print(f"全部处理完成！最终输出文件: {OUTPUT_FILE}，共 {len(existing_entries)} 条记录。")

if __name__ == "__main__":
    main()