import os
from http import HTTPStatus

import dashscope
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import embedding_model_loader
from langchain_base_document_loader import load_document

os.environ["TOKENIZERS_PARALLELISM"] = "false"

qwen_model = "qwen-turbo"
qwen_api_key = ""

def scan_document_folder(document_folder):
    documents = []

    for file_name in os.listdir(document_folder):
        file_path = os.path.join(document_folder, file_name)

        # 递归扫描文件
        if os.path.isdir(file_path):
            documents.extend(scan_document_folder(file_path))
            continue

        elif not os.path.isfile(file_path):
            continue

        print(f"加载文件 {file_path}")

        try:
            documents.append(load_document(file_path))
        except ValueError as e:
            print(f"文件 {file_path} 跳过，错误信息：{e}")

    return documents

def indexing_process(documents, empbedding_model):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    embeddings = []
    all_chunks = []

    for document in documents:
        chunks = text_splitter.split_text(document)
        all_chunks.extend(chunks)

        for chunk in chunks:
            embedding = empbedding_model.encode(chunk)
            embeddings.append(embedding)

    embeddings_np = np.array(embeddings)
    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    return index, all_chunks


def retrieval_process(query, index, chunks, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, top_k)

    print(f"查询语句 {query}")
    print(f"最相似的前{top_k}个文本块")

    results = []

    for i in range(top_k):
        result_chunk = chunks[indices[0][i]]

        print(f"第{i+1}个文本块 {result_chunk}\r\n")
        print(f"相似度：{distances[0][i]}\r\n")

        results.append(chunks[indices[0][i]])

    return results

def generate_process(query, chunks):
    llm_model = qwen_model
    dashscope.api_key = qwen_api_key

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}:\n{chunk}\n\n"

    prompt = f"根据参考文档回答问题:{query}\n\n{context}"
    print(f"prompt:{prompt}")

    messages = [{"role": "user", "content": prompt}]

    try:
        responses = dashscope.Generation.call(
            model=llm_model,
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True
        )

        generate_response = ""
        print("开始生成：")
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                generate_response += content
                print(content, end="")
            else:
                print(f"Error: {response.message}")
                return None

        print("\n生成结束")
        return generate_response
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("RAG过程开始.")

    global qwen_api_key
    qwen_api_key = os.getenv("qwen_key")
    if qwen_api_key is None:
        print("请设置qwen_key环境变量")
        return

    query="怎么查看当前系统中的所有pv"
    embedding_model = embedding_model_loader.load_embedding_model()

    documents = scan_document_folder('E:/Document/notes_collation/linux')
    if len(documents) == 0:
        print("没有需要处理的文件")
        return

    # 索引流程：加载PDF文件，分割文本块，计算嵌入向量，存储在FAISS向量库中（内存）
    index, chunks = indexing_process(documents, embedding_model)

    # 检索流程：将用户查询转化为嵌入向量，检索最相似的文本块
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # 生成流程：调用Qwen大模型生成响应
    # generate_process(query, retrieval_chunks)

    print("RAG过程结束.")

if __name__ == "__main__":
    main()