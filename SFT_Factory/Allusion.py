import json
import os
import time
import random
import signal
import sys
from openai import OpenAI

# ================= 配置 =================
# 111111111111111111111111111111111
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-0c3f93263afd4877a422690219514805")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"

FOLDER_PATH = r"/data"
OUTPUT_FILE = os.path.join(FOLDER_PATH, "../SFT_DataSet/allusions_comprehension.json")
PROGRESS_FILE = os.path.join(FOLDER_PATH, "allusion_progress.json")

BATCH_SIZE = 5000
TARGET_ALLUSION_COUNT = 300
SLEEP_BETWEEN_CALLS = 0.5

SYSTEM_PROMPT_STEP2 = (
    "你是中国古典诗词赏析专家，能精准识别诗词中的典故。"
    "回答时请先给出推理过程，再给出最终答案。"
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 全局变量，用于中断时保存状态
should_exit = False


def signal_handler(sig, frame):
    global should_exit
    print("\n\n收到中断信号，正在保存当前进度...")
    should_exit = True


signal.signal(signal.SIGINT, signal_handler)


def extract_paragraphs_from_json(data):
    paragraphs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "paragraphs" in item:
                para_list = item["paragraphs"]
                if isinstance(para_list, list):
                    paragraphs.extend(para_list)
                elif isinstance(para_list, str):
                    paragraphs.append(para_list)
    elif isinstance(data, dict):
        if "paragraphs" in data:
            para_list = data["paragraphs"]
            if isinstance(para_list, list):
                paragraphs.extend(para_list)
            elif isinstance(para_list, str):
                paragraphs.append(para_list)
    return paragraphs


def check_has_allusion(paragraph: str) -> bool:
    user_prompt = f"请你用“是”或“否”，直接回答诗句“{paragraph}”中是否含有典故？"
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        if "是" in answer and "否" not in answer:
            return True
        return False
    except Exception as e:
        print(f"API 调用失败（典故判断）：{e}")
        return False


def get_allusion_explanation(paragraph: str) -> str:
    user_prompt = f"诗句“{paragraph}”中引用了什么典故？"
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_STEP2},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 调用失败（典故解释）：{e}")
        return None


def save_output(records):
    """全量写入输出文件"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found):
    """保存进度状态"""
    progress = {
        "processed_indices": list(processed_indices),
        "all_paragraphs": all_paragraphs,  # 保存段落原文，避免重复加载
        "shuffled_indices": shuffled_indices,
        "allusions_found": allusions_found,
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_progress():
    """加载进度，返回 (output_records, processed_indices, all_paragraphs, shuffled_indices, allusions_found)"""
    if not os.path.exists(OUTPUT_FILE) or not os.path.exists(PROGRESS_FILE):
        return [], set(), None, None, 0

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_records = json.load(f)
    except:
        output_records = []

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        processed_indices = set(progress.get("processed_indices", []))
        all_paragraphs = progress.get("all_paragraphs", [])
        shuffled_indices = progress.get("shuffled_indices", [])
        allusions_found = progress.get("allusions_found", 0)
        return output_records, processed_indices, all_paragraphs, shuffled_indices, allusions_found
    except:
        return output_records, set(), None, None, 0


def main():
    global should_exit

    # 加载已有进度
    output_records, processed_indices, existing_paragraphs, existing_shuffled, allusions_found = load_progress()
    print(f"加载进度：已有典故条目 {len(output_records)} 条，已处理段落 {len(processed_indices)} 个")

    # 如果需要从头开始或进度文件不完整，重新读取所有段落
    if existing_paragraphs is None or len(existing_paragraphs) == 0:
        if not os.path.isdir(FOLDER_PATH):
            print(f"错误：文件夹 '{FOLDER_PATH}' 不存在")
            return

        # 筛选文件名包含 'poet' 或 'ci' 的 json 文件
        json_files = []
        for f in os.listdir(FOLDER_PATH):
            if not f.lower().endswith(".json"):
                continue
            lower_name = f.lower()
            if "poet" in lower_name or "ci" in lower_name:
                json_files.append(f)

        if not json_files:
            print(f"警告：文件夹 '{FOLDER_PATH}' 中没有找到文件名包含 'poet' 或 'ci' 的 JSON 文件")
            return

        print(f"找到符合条件的 JSON 文件：{json_files}")

        all_paragraphs = []
        for file_name in json_files:
            file_path = os.path.join(FOLDER_PATH, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                paras = extract_paragraphs_from_json(data)
                print(f"从 {file_name} 中提取到 {len(paras)} 个段落")
                all_paragraphs.extend(paras)
            except Exception as e:
                print(f"读取文件 {file_name} 失败：{e}")

        total_paragraphs = len(all_paragraphs)
        print(f"总共提取到 {total_paragraphs} 个段落")

        if total_paragraphs == 0:
            print("没有可处理的段落，退出。")
            return

        # 随机打乱索引
        random.seed(64)
        shuffled_indices = list(range(total_paragraphs))
        random.shuffle(shuffled_indices)

        # 保存初始状态
        save_progress(set(), all_paragraphs, shuffled_indices, 0)
    else:
        all_paragraphs = existing_paragraphs
        shuffled_indices = existing_shuffled
        print(f"恢复已有段落总数：{len(all_paragraphs)}")

    total_paragraphs = len(all_paragraphs)

    # 确定未处理的索引列表
    remaining_indices = [idx for idx in shuffled_indices if idx not in processed_indices]
    print(f"剩余未处理段落数：{len(remaining_indices)}")

    if allusions_found >= TARGET_ALLUSION_COUNT:
        print(f"已达到目标 {TARGET_ALLUSION_COUNT} 条典故，无需继续。")
        return

    # 开始处理
    processed_this_run = 0
    for idx in remaining_indices:
        if should_exit or allusions_found >= TARGET_ALLUSION_COUNT:
            break

        para = all_paragraphs[idx]
        if not para or not para.strip():
            # 空段落也标记为已处理
            processed_indices.add(idx)
            save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found)
            continue

        processed_this_run += 1
        print(f"\n[剩余 {len(remaining_indices)-processed_this_run+1} 个未处理] 正在分析：{para[:80]}...")

        # 第一步：判断是否有典故
        has_allusion = check_has_allusion(para)
        if not has_allusion:
            print("  结果：无典故（跳过）")
            processed_indices.add(idx)
            save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found)
            continue

        # 第二步：获取典故解释
        print("  结果：有典故，正在获取解释...")
        explanation = get_allusion_explanation(para)
        if explanation is None:
            print("  获取解释失败，跳过")
            processed_indices.add(idx)
            save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found)
            continue

        # 保存结果
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP2},
            {"role": "user", "content": f"诗句“{para}”中引用了什么典故？"},
            {"role": "assistant", "content": explanation},
        ]
        output_records.append({"messages": messages})
        allusions_found += 1
        print(f"  已保存典故（总数：{allusions_found}）")
        print(f"  解释内容：{explanation[:200]}...")

        # 立即写入输出文件和进度文件
        save_output(output_records)
        processed_indices.add(idx)
        save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found)

        if allusions_found >= TARGET_ALLUSION_COUNT:
            print(f"\n已达到目标 {TARGET_ALLUSION_COUNT} 条典故，停止处理。")
            break

        time.sleep(SLEEP_BETWEEN_CALLS)

    # 最终保存
    save_output(output_records)
    save_progress(processed_indices, all_paragraphs, shuffled_indices, allusions_found)

    print("\n=== 处理完成 ===")
    print(f"本次运行处理新段落数：{processed_this_run}")
    print(f"最终典故条目数：{allusions_found}")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"进度文件：{PROGRESS_FILE}")


if __name__ == "__main__":
    main()