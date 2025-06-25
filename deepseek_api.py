from openai import OpenAI
import json
from redis_top5 import VectorSearchEvaluator
from build_prompt import build_prompt_with_rag  # 用你自己的 prompt 构建函数
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('BAAI/bge-m3')

# 初始化向量搜索模块
retriever = VectorSearchEvaluator()

# 文件路径
file = "QAdata.json"
output_file = "output_full.json"  # 存储完整的问答与推理过程
similarity_file = "output_similarity.json"  # 存储题目序号与相似度得分

# 加载数据
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]
answer_types = [item['answer_type'] for item in data]
evi_paras = [item['evidence_paragraphs'] for item in data]

# 初始化用于存储结果的列表
full_results = []
similarity_results = []

# 遍历所有数据
for i in range(len(data)):

    # 用户提问
    query = questions[i]
    print(f"用户提问：{query}")

    # 检索相关段落（只拿段落文本）
    retrieved_docs = retriever.search_top_k(query)
    retrieved_texts = [
        f"[段落 {item['para_id'].split(':')[-1]}]\n{item['text']}"
        for item in retrieved_docs
    ]

    if i < 5:
        print("检测到的 Top5 相关文档段落：\n")
        for j, item in enumerate(retrieved_docs):
            para_id = item["para_id"].split(":")[-1]
            print(f"Top{j+1} - 原始段落编号: {para_id}")
            print(f"{item['text']}\n")

    # 构造 Prompt —— 使用你自己的函数（build_prompt_with_rag）
    prompt = build_prompt_with_rag(query, retrieved_texts)

    # 调用 DeepSeek-V3 模型（OpenAI SDK 兼容格式）
    def call_deepseek_v3_with_custom_prompt(prompt: str):
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",  # DeepSeek V3 国内API
            api_key="your_api_key_here"  # 替换为你的 API Key
        )

        response = client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    answerwithCoT = call_deepseek_v3_with_custom_prompt(prompt)

    # 提取 ### 最终答案: 后面的部分（非贪婪匹配，避免截取过多）
    match = re.search(r"### 最终答案:\s*(.+)", answerwithCoT, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        answer = "未能识别最终答案，请检查模型输出格式。"
    print(f'最终答案：{answer}')

    # 计算余弦相似度
    ans_em = model.encode(answer)
    std_em = model.encode(answers[i])
    cosine_sim = np.dot(ans_em, std_em) / (np.linalg.norm(ans_em) * np.linalg.norm(std_em))
    print("与原答案的余弦相似度:", cosine_sim)

    # 保存完整结果
    full_results.append({
        "question": query,
        "retrieved_texts": retrieved_texts,
        "answer_with_CoT": answerwithCoT,
        "final_answer": answer,
        "cosine_similarity": cosine_sim
    })

    # 保存相似度结果
    similarity_results.append({
        "question_id": i + 1,  # 题目序号（从1开始）
        "cosine_similarity": cosine_sim
    })

# 将完整的问答与推理过程保存为文档
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(full_results, f, ensure_ascii=False, indent=4)

# 将题目序号和相似度得分保存为文档
with open(similarity_file, 'w', encoding='utf-8') as f:
    json.dump(similarity_results, f, ensure_ascii=False, indent=4)

# 计算相似度的平均值
average_similarity = np.mean([result["cosine_similarity"] for result in similarity_results])
print(f"所有问答的相似度平均值: {average_similarity}")