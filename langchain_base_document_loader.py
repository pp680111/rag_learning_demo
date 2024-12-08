import os.path

from langchain_community.document_loaders import ( PDFPlumberLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader, CSVLoader, UnstructuredMarkdownLoader, UnstructuredXMLLoader, UnstructuredHTMLLoader,)

DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PDFPlumberLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docs": (UnstructuredWordDocumentLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xml": (UnstructuredXMLLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}

def load_document(file_path):
    if not file_path:
        raise ValueError("File path is empty.")

    ext = os.path.splitext(file_path)[1]
    loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext.lower())
    if not loader_tuple:
        raise ValueError(f"Unsupported file type: {ext}")

    loader_class, loader_arg = loader_tuple
    loader = loader_class(file_path, **loader_arg)

    documents = loader.load()

    content = "\n".join(([doc.page_content for doc in documents]))
    # print(f"文档内容：{content[:100]}...\r\n")

    return content