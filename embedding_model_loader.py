import os

from sentence_transformers import SentenceTransformer


def load_embedding_model():
    print(f"加载Embedding模型中")
    # SentenceTransformer是一个用来加载Embedding模型的库，它本身特提供了从huggingface下载模型的功能
    embedding_model = SentenceTransformer(os.path.abspath("bge-small-zh-v1.5"))
    # print(f"bge-large-zh-v1.5模型最大输入长度:{embedding_model.max_seq_length}")
    return embedding_model
