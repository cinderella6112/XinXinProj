import json
import random
from typing import Dict, List, Any

# ================== 系统提示词模板 ==================
SYSTEM_PROMPTS = {
    "word": "你是中国古典诗词赏析专家，精通字词训诂。回答时请先给出推理过程，再给出最终答案。",
    "sentence": "你是中国古典诗词赏析专家，擅长将古诗译为准确流畅的现代汉语。回答时请先给出推理过程，再给出最终答案。",
    "emotion": "你是中国古典诗词赏析专家，能精准把握诗人情感。回答时请先给出推理过程，再给出最终答案。",
    "allusion": "你是中国古典诗词赏析专家，精通典故溯源。回答时请先判断是否用典，再解释典故内容及用意。",
    "analogy": "你是中国古典诗词赏析专家，擅长发现诗句中的意象关系。回答时请先分析关系，再给出最终答案。",
    "discrimination": "你是中国古典诗词赏析专家，能准确辨析诗句的理解与赏析。回答时请逐项分析，再给出最终答案。"
}

# ================== 辅助函数 ==================
def format_answer(reasoning: str, answer: str) -> str:
    """统一格式化答案，强制包含推理过程和最终答案"""
    return f"推理过程：{reasoning}\n最终答案：{answer}"

def generate_distractor_options(correct_emotion: str, emotion_pool: List[str] = None) -> List[str]:
    """为情感选择题生成干扰项"""
    if emotion_pool is None:
        emotion_pool = ["喜悦", "悲伤", "愤懑", "闲适", "思乡", "怀古", "壮志", "惆怅", "孤独", "豪迈"]
    # 去除正确情感，随机选3个作为干扰项
    candidates = [e for e in emotion_pool if e != correct_emotion]
    distractors = random.sample(candidates, min(3, len(candidates)))
    options = distractors + [correct_emotion]
    random.shuffle(options)
    return options

def build_messages(system: str, user: str, assistant: str) -> Dict:
    """构建标准的 messages 格式"""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

# ================== task1.json 转换 ==================
def convert_task1(item: Dict) -> List[Dict]:
    samples = []
    title = item.get("title", "无题")
    content = item.get("content", "")
    keywords = item.get("keywords", {})
    trans = item.get("trans", "")
    emotion = item.get("emotion", "")

    # 1. 字词理解（只包含目标诗句）
    lines = [line.strip() + "。" for line in content.split("。") if line.strip()]
    for word, meaning in keywords.items():
        target_line = ""
        for line in lines:
            if word in line:
                target_line = line
                break
        if not target_line:
            target_line = content  # 兜底

        user = f"在诗句“{target_line}”中，“{word}”是什么意思？"
        assistant = format_answer(
            reasoning=f"“{word}”在此处指{meaning}，结合诗句语境可确定其含义。",
            answer=meaning
        )
        samples.append(build_messages(SYSTEM_PROMPTS["word"], user, assistant))

    # 2. 诗句理解（翻译全诗，评测常见）
    if trans:
        user = f"请将以下诗句翻译成现代汉语：\n{content}"
        assistant = format_answer(
            reasoning="逐句理解诗意，用流畅的现代汉语表达。",
            answer=trans
        )
        samples.append(build_messages(SYSTEM_PROMPTS["sentence"], user, assistant))

    # 3. 情感理解（全诗，构造选择题）
    if emotion:
        options = generate_distractor_options(emotion)
        option_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        correct_letter = chr(65 + options.index(emotion))
        user = f"阅读诗句：“{content}”\n这首诗表达了诗人怎样的情感？\n{option_str}"
        assistant = format_answer(
            reasoning=f"从诗歌意象、关键词及整体氛围来看，诗人主要表达了{emotion}的情感。",
            answer=correct_letter
        )
        samples.append(build_messages(SYSTEM_PROMPTS["emotion"], user, assistant))

    return samples

# ================== task2.json 转换 ==================
def convert_task2(item: Dict) -> Dict:
    """
    处理 task2.json 中的单条数据
    字段：que, answer
    生成：典故识别 / 字词理解样本
    """
    que = item.get("que", "")
    answer = item.get("answer", "")

    user = f"诗句“{que}”中包含什么典故或文化知识？请解释。"
    assistant = format_answer(
        reasoning="分析诗句中的关键词，追溯其历史或文学典故。",
        answer=answer
    )
    return build_messages(SYSTEM_PROMPTS["allusion"], user, assistant)

# ================== task3.json 转换（可选） ==================
def convert_task3(item: Dict) -> Dict:
    """
    处理 task3.json 中的单条数据（诗词填空）
    由于评测不直接考填空，此处少量用于构造“古诗词类比”任务
    """
    que = item.get("que", "")
    answers = item.get("answer", [])

    # 简单示例：将填空转换为类比题（实际可更精细设计）
    user = f"在古诗词中，常有对仗工整的上下句。例如“无边落木萧萧下”与“不尽长江滚滚来”。下列选项中，哪一组关系与此最相似？\nA. {answers[0] if len(answers)>0 else ''} / {answers[1] if len(answers)>1 else ''}\nB. ... (此处需手工完善)"
    # 由于 task3 结构特殊，建议仅抽取少量手工处理，此处略
    return None

# ================== task4.json 转换 ==================
def convert_task4(item: Dict) -> List[Dict]:
    """
    处理 task4.json 中的单条数据
    字段：title, author, content, zhushi, questions (含 options 和 answer)
    生成：古诗词辨析样本
    """
    samples = []
    title = item.get("title", "")
    author = item.get("author", "")
    content = item.get("content", "")
    zhushi = item.get("zhushi", [])

    for q in item.get("questions", []):
        que = q.get("que", "")
        options = q.get("options", {})
        answer = q.get("answer", "")

        # 构造选项文本
        option_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

        user = f"阅读下面这首诗，回答问题。\n\n《{title}》\n{author}\n{content}\n"
        if zhushi:
            user += "\n注释：" + "；".join(zhushi)
        user += f"\n\n{que}\n{option_str}"

        # 生成推理过程（可基于正确选项描述）
        correct_reason = options.get(answer, "")
        assistant = format_answer(
            reasoning=f"逐项分析各选项，{answer}项所述符合诗歌原意，其他选项存在理解偏差。",
            answer=answer
        )
        samples.append(build_messages(SYSTEM_PROMPTS["discrimination"], user, assistant))

    return samples

# ================== 主转换流程 ==================
def main():
    all_samples = []

    # 1. 处理 task1.json
    with open("task1.json", "r", encoding="utf-8") as f:
        task1_data = json.load(f)
    for item in task1_data:
        samples = convert_task1(item)
        all_samples.extend(samples)
    print(f"task1 生成 {len([s for item in task1_data for s in convert_task1(item)])} 条样本")

    # 2. 处理 task2.json
    with open("task2.json", "r", encoding="utf-8") as f:
        task2_data = json.load(f)
    for item in task2_data:
        sample = convert_task2(item)
        all_samples.append(sample)
    print(f"task2 生成 {len(task2_data)} 条样本")

    # 3. 处理 task4.json
    with open("task4.json", "r", encoding="utf-8") as f:
        task4_data = json.load(f)
    for item in task4_data:
        samples = convert_task4(item)
        all_samples.extend(samples)
    print(f"task4 生成 {sum(len(convert_task4(item)) for item in task4_data)} 条样本")

    # 4. 可选：加入少量通用对话样本（防灾难性遗忘）
    general_samples = [
        build_messages(
            "你是古诗词赏析助手，请用现代汉语友好回应。",
            "你好，请介绍一下你自己。",
            "你好，我是古诗词赏析助手，可以帮你解答关于古诗的字词、句意、情感和典故问题。"
        ),
        build_messages(
            "你是古诗词赏析助手。",
            "今天天气真好。",
            "是啊，风和日丽，不禁让人想起“阳春布德泽，万物生光辉”。需要我为您赏析一首春日古诗吗？"
        )
    ]
    all_samples.extend(general_samples)

    # 5. 保存为 JSONL
    output_file = "poetry_sft_data.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"转换完成！共生成 {len(all_samples)} 条 SFT 样本，已保存至 {output_file}")

if __name__ == "__main__":
    main()