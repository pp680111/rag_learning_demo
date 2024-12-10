import os.path

import chromadb
import uuid
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 要注意，高版本的chromadb在windows下运行collection.add方法会崩溃，所以这里推荐用0.5.3版本的
_chroma_db_path = os.path.abspath("chroma_db")
print(f"chroma db init, file path = {_chroma_db_path}")

# # 启动时删除旧数据库文件，这里看着注释
# if os.path.exists(_chroma_db_path):
#     shutil.rmtree(_chroma_db_path)

_client = chromadb.PersistentClient(path=_chroma_db_path)
_collection = _client.get_or_create_collection(name="my_collection")


def add_documents(documents, embedding_model):
    """
    添加新的文档的向量
    :param documents:
    :param embedding_model:
    :return:
    """

    all_chunks = []
    all_ids = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    for doc in documents:
        chunks = text_splitter.split_text(doc)

        # 每一个chunk都生成一个uuid
        all_chunks.extend(chunks)
        all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    print(f"已处理chunk数{len(all_chunks)}")

    # embedding_model生成的向量是numpy的ndarray，但是chromadb要求的是一个list
    embeddings = [embedding_model.encode(chunk, normalize_embeddings=True).tolist() for chunk in all_chunks]

    _collection.add(ids = all_ids, embeddings = embeddings, documents = all_chunks)

def retrieval_process(query, embedding_model=None, top_k = 6):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = _collection.query(query_embeddings=[query_embedding], n_results=top_k)

    print(f"查询到的最相似的前{top_k}个文本块：")

    retrieved_chunks = []
    for doc_id, doc, score in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
        print(f"文本块ID: {doc_id}")
        print(f"相似度: {score}")
        print(f"文本块信息:\n{doc}\n")
        retrieved_chunks.append(doc)

    return retrieved_chunks
