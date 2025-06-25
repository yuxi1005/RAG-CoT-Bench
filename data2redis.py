import pandas as pd
import numpy as np
import redis
import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# 加载其他模型前先清除索引：FT.DROPINDEX doc_idx DD
model = SentenceTransformer("BAAI/bge-base-zh")
VECTOR_DIM = 768

# Redis连接
r = redis.Redis(host="192.168.0.207", port=6380)

# 创建 Redis 向量索引
try:
    r.execute_command(
        f"FT.CREATE doc_idx ON HASH PREFIX 1 doc SC:HEMA text TEXT vec VECTOR HNSW 6 TYPE FLOAT32 DIM {VECTOR_DIM} DISTANCE_METRIC COSINE"
    )
except redis.exceptions.ResponseError as e:
    if "Index already exists" not in str(e):
        raise e

# =================== TXT 分段数据处理函数 ====================
def process_txt_file(file_path, min_length=10, max_length=301):
    """
    处理形如 "[段落 1] 内容" 的分段文本文件，并写入 Redis 向量库。
    - file_path: TXT 文件路径
    - min_length: 段落最小字数限制（小于该值将被忽略）
    - max_length: 段落最大字数限制（超过该值将被忽略）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取形如：[段落 1] 段落内容
    pattern = r"\[段落 (\d+)\]\s*(.*?)\n(?=\[段落|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    # 仅保留字数在指定范围内的段落
    paragraphs = [
        (int(num), text.strip())
        for num, text in matches
        if min_length <= len(text.strip()) <= max_length
    ]

    if not paragraphs:
        print("⚠️ 未找到合规段落（长度需在指定范围内）。")
        return

    texts = [text for _, text in paragraphs]
    embeddings = model.encode(texts, normalize_embeddings=True)

    for (pid, text), vec in zip(paragraphs, embeddings):
        key = f"doc:{pid}"
        r.hset(key, mapping={
            "text": text,
            "vec": vec.astype(np.float32).tobytes()
        })

    print(f"✅ TXT 文件处理完成，共写入 {len(paragraphs)} 个段落")
# ====================================================================


# 处理 TXT 文件---长文档&短查询任务
data1 = "ChangAnLiZhi.txt"
process_txt_file(data1)
print(f"✅长文本{data1}已写入")