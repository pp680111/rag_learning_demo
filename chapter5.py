import os
from http import HTTPStatus

import dashscope

import embedding_model_loader
import vector_db
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
    #
    # documents = scan_document_folder('E:/Document/notes_collation/linux')
    # if len(documents) == 0:
    #     print("没有需要处理的文件")
    #     return
    #
    # vector_db.add_documents(documents, embedding_model)
    print(f"处理文本索引完毕，开始执行查询")
    vector_db.retrieval_process(query, embedding_model=embedding_model, top_k=3)

    # 生成流程：调用Qwen大模型生成响应
    # generate_process(query, retrieval_chunks)

    print("RAG过程结束.")

if __name__ == "__main__":
    main()