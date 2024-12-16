import jieba
from rank_bm25 import BM25Okapi


def search_by_keyword(query, documents, top_k=6):
    # 对搜索语句和所有文档进行中文切词
    tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
    tokenized_query = list(jieba.cut(query))

    # 使用bm25检索算法，获取搜索语句在每个文档中的匹配得分
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)

    # 按检索得分
    bm25_top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_chunks = [documents[i] for i in bm25_top_k_indices]

    for i in range(len(bm25_chunks)):
        print(f"关键字检索排名:{i}, chunk:{bm25_chunks[i]}\r\n")

    return bm25_chunks
