# embedding_gen.py
# 生成《长安的荔枝》每段的BGE-M3向量，并保存为JSON

from sentence_transformers import SentenceTransformer
import json

# === 配置 ===
MODEL_NAME = "BAAI/bge-m3"
INPUT_FILE = "data/ChangAnLiZhi.txt"  # 每行格式：[段落编号]正文内容
OUTPUT_FILE = "data/ChangAnLiZhi_with_embeddings.json"

# === 加载模型 ===
model = SentenceTransformer(MODEL_NAME)
print("✅ BGE-M3 模型加载完成")

# === 读取段落文本 ===
paragraphs = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith("[段落"):
            continue
        try:
            idx_end = line.index("]")
            chunk_id = int(line[3:idx_end])
            text = line[idx_end + 1:].strip()
            paragraphs.append({"chunk_id": chunk_id, "text": text})
        except:
            print(f"❌ 跳过异常行: {line}")

print(f"📚 共读取到 {len(paragraphs)} 个段落")

# === 提取文本并生成向量 ===
texts = [p["text"] for p in paragraphs]
embeddings = model.encode(texts, normalize_embeddings=True)

# === 合并结果并写入 JSON ===
for i in range(len(paragraphs)):
    paragraphs[i]["embedding"] = embeddings[i].tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(paragraphs, f, ensure_ascii=False, indent=2)

print(f"✅ 向量写入成功，输出文件：{OUTPUT_FILE}")
