import redis
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer


class VectorSearchEvaluator:
    def __init__(self, host="192.168.0.207", port=6380, model_name="BAAI/bge-base-zh"):
        """初始化Redis连接和Sentence Transformer模型"""
        self.r = redis.Redis(host=host, port=port)
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text, **kwargs):
        return self.model.encode(text, **kwargs)


    def search_top_k(self, query_text, top_k=5):
        """
        输入查询句，返回TopK段落
        :param query_text: 查询文本
        :param top_k: 返回结果数量
        :return: List[Dict]，每项包含段落编号和段落内容
        """
        query_vec = self.encode_text(query_text, normalize_embeddings=True)
        query_bytes = np.array(query_vec, dtype=np.float32).tobytes()
        try:
            res = self.r.execute_command(
                "FT.SEARCH",
                "doc_idx",
                f"*=>[KNN {top_k} @vec $vec AS score]",
                "PARAMS", "2", "vec", query_bytes,
                "DIALECT", "2"
            )
        except redis.exceptions.ResponseError as e:
            print(f"查询失败：{e}")
            return []

        results = []
        if len(res) <= 1:
            return results

        # 解析返回结果，格式为 [总数, doc_id1, fields1, doc_id2, fields2, ...]
        for i in range(1, len(res), 2):
            doc_id = res[i].decode() if isinstance(res[i], bytes) else res[i]
            fields_list = res[i + 1]
            text = "[无 text 字段]"
            for k, v in zip(fields_list[::2], fields_list[1::2]):
                key = k.decode() if isinstance(k, bytes) else str(k)
                if key == "text":
                    text = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            results.append({
                "para_id": doc_id,
                "text": text
            })
        return results


# if __name__ == "__main__":
#     evaluator = VectorSearchEvaluator()
#     test_query = "荔枝从岭南运到长安需要几天？"
#     top_k_results = evaluator.search_top_k(test_query, top_k=5)


#     print("\n查询结果 Top5：")
#     for i, para in enumerate(top_k_results):
#         print(f"[{i+1}] 段落编号: {para['para_id']}")
#         print(f"     内容：{para['text'][:80]}...\n")
