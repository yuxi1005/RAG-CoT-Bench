def build_prompt_with_rag(query: str, contexts: list[str]) -> str:
    context_text = "\n".join([f"[段落{i+1}]\n{para}" for i, para in enumerate(contexts)])

    return f"""你是一位擅长文档阅读与逻辑推理的大模型助手，以下是来自一篇文献《长安的荔枝》的若干段落，请你基于这些段落，按照“逐步思考”与“明确作答”两个部分完成回答。

【背景资料】
{context_text}

【用户问题】
{query}

【作答要求】
- 所有推理必须严格基于提供的背景资料，不得引入外部常识或假设
- 回答需分为两个部分：
  1. 【思考过程】：逐步分析并串联段落内容，展示你的推理路径
  2. `### 最终答案:` 在此明确回答用户问题，内容应简洁、具体、可直接引用
- 严格使用以上格式标注，确保输出规范

【回答】
【思考过程】
（请在此展开逐步推理……）

### 最终答案:
（请在此明确回答用户问题）
"""
