import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI

# 初始化DeepSeek客户端，兼容OpenAI库
client = OpenAI(
    api_key="sk-0c3f93263afd4877a422690219514805", # 替换成你的API Key
    base_url="https://api.deepseek.com",
)

def call_deepseek(prompt: str, max_tokens: int = 2048) -> Optional[Dict]:
    """调用DeepSeek Reasoner模型，返回包含推理过程和答案的字典"""
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        # R1模型的响应包含 reasoning_content 和 content 字段
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', '')
        answer = message.content or ''
        return {"reasoning": reasoning, "answer": answer}
    except Exception as e:
        print(f"API调用失败: {e}")
        return None

def generate_word_qa(poem_info: Dict) -> List[Dict]:
    """生成字词理解任务的数据 (task1)"""
    samples = []
    title, content, keywords = poem_info["title"], poem_info["content"], poem_info["keywords"]
    lines = [line.strip() + "。" for line in content.split("。") if line.strip()]
    for word, meaning in keywords.items():
        # 定位包含词语的诗句
        target_line = next((line for line in lines if word in line), content)
        prompt = f"""你是一位古诗词专家。请为以下字词理解任务生成一个问答对。
【诗词】《{title}》
【目标诗句】{target_line}
【词语】{word}
【参考释义】{meaning}
【要求】
1. 生成一个用户提问，询问该词语的含义，例如“在诗句‘{target_line}’中，‘{word}’是什么意思？”
2. 生成答案，必须包含“推理过程：...\n最终答案：...”两部分。推理过程需结合诗句语境分析词语的准确含义。
3. 最终答案部分请使用参考释义，但可以根据语境进行微调。
请将结果按JSON格式输出：{{"question": "...", "answer": "..."}}"""
        response = call_deepseek(prompt)
        if response:
            # 清理可能多余的空白或Markdown标记
            answer_text = response["answer"].strip()
            if answer_text.startswith("```json"):
                answer_text = answer_text[7:]
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]
            try:
                qa_data = json.loads(answer_text)
                samples.append({
                    "messages": [
                        {"role": "system", "content": "你是中国古典诗词赏析专家，精通字词训诂。"},
                        {"role": "user", "content": qa_data["question"]},
                        {"role": "assistant", "content": qa_data["answer"]}
                    ]
                })
            except json.JSONDecodeError:
                print(f"JSON解析失败，原始输出: {response['answer']}")
        time.sleep(1) # 避免请求过快
    return samples

def generate_allusion_qa(poem_info: Dict) -> Optional[Dict]:
    """生成典故识别任务的数据 (task2, task4)"""
    title, author, content, zhushi = poem_info["title"], poem_info["author"], poem_info["content"], poem_info.get("zhushi", [])
    prompt = f"""你是一位中国古典诗词专家。请根据下面提供的诗词信息，完成“典故识别”任务。
【诗词信息】标题：{title}\n作者：{author}\n内容：{content}\n注释：{zhushi}
【任务要求】
1. 判断这首诗词是否使用了典故。如果没有，回答“无”。
2. 如果使用了典故，请生成一个问答对，格式如下：
   - 问题：针对诗词中的某个典故提问，例如“诗句‘{content[:20]}...’中引用了什么典故？请解释其含义和在诗中的作用。”
   - 答案：必须包含“推理过程：...\n最终答案：...”。推理过程需结合诗词背景和注释，详细解释典故的出处、原意以及在诗中的寓意。
请直接输出JSON格式：{{"has_allusion": true/false, "question": "...", "answer": "..."}}"""
    response = call_deepseek(prompt)
    if response:
        answer_text = response["answer"].strip()
        if answer_text.startswith("```json"): answer_text = answer_text[7:]
        if answer_text.endswith("```"): answer_text = answer_text[:-3]
        try:
            qa_data = json.loads(answer_text)
            if qa_data.get("has_allusion"):
                return {
                    "messages": [
                        {"role": "system", "content": "你是中国古典诗词赏析专家，精通典故溯源。"},
                        {"role": "user", "content": qa_data["question"]},
                        {"role": "assistant", "content": qa_data["answer"]}
                    ]
                }
        except json.JSONDecodeError:
            print(f"JSON解析失败，原始输出: {response['answer']}")
    return None

# 主函数
def main():
    all_samples = []
    # 处理 task1 数据
    with open("task1.json", "r", encoding="utf-8") as f:
        task1_data = json.load(f)
    for item in task1_data[:2]: # 为了快速演示，只处理前2条，可以删除切片
        all_samples.extend(generate_word_qa(item))
    # 处理 task2 数据
    with open("task2.json", "r", encoding="utf-8") as f:
        task2_data = json.load(f)
    for item in task2_data[:2]:
        # 将task2的que和answer字段构造成类似诗词信息
        sample = generate_allusion_qa({"title": "", "author": "", "content": item["que"], "zhushi": [item["answer"]]})
        if sample: all_samples.append(sample)
    # 保存结果
    with open("sft_data_from_api.jsonl", "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"成功生成 {len(all_samples)} 条SFT数据，已保存至 sft_data_from_api.jsonl")

if __name__ == "__main__":
    main()