from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import dashscope
from http import HTTPStatus

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

qwen_model = "qwen-turbo"
qwen_api_key = "sk-cb061839ce3c489886dbaac15b46396d"

def load_embedding_model():
    print(f"加载Embedding模型中")
    embedding_model = SentenceTransformer(os.path.abspath("bge-small-zh-v1.5"))
    print(f"bge-small-zh-v1.5模型最大输入长度:{embedding_model.max_seq_length}")
    return embedding_model

def indexing_process(pdf_file, empbedding_model):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    pdf_content_list = pdf_loader.load()
    pdf_text = "\n".join([pdf_content.page_content for pdf_content in pdf_content_list])
    print(f"PDF总字符数：{len(pdf_text)}")

    chunks = text_splitter.split_text(pdf_text)
    print(f"PDF文本分片数：{len(chunks)}")

    embeddings = []
    for chunk in chunks:
        embedding = empbedding_model.encode(chunk)
        embeddings.append(embedding)

    print("已完成文本的向量化")

    embeddings_np = np.array(embeddings)

    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    return index, chunks


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

    query="下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    embedding_model = load_embedding_model()

    # 索引流程：加载PDF文件，分割文本块，计算嵌入向量，存储在FAISS向量库中（内存）
    index, chunks = indexing_process('test_lesson2.pdf', embedding_model)

    # 检索流程：将用户查询转化为嵌入向量，检索最相似的文本块
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # 生成流程：调用Qwen大模型生成响应
    generate_process(query, retrieval_chunks)

    print("RAG过程结束.")

if __name__ == "__main__":
    main()