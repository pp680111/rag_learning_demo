import langchain_base_document_loader

def load_document_test():
    content = langchain_base_document_loader.load_document('E:/Document/笔记整理/linux/fork.md')
    print(content)

load_document_test()