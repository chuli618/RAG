from query_processor import QueryProcessor
from query_context import QueryContext
import os
import PyPDF2

class DocSplit(QueryProcessor):
    """PDF文件分块处理器"""
    
    def __init__(self):
        """初始化PDF分块处理器"""
        super().__init__()
        self.name = "DocSplit"

    def process(self, context: QueryContext):
        """
        处理PDF文件并提取文本内容。

        参数:
            context (QueryContext): 包含处理所需参数的上下文对象

        返回:
            None: 该方法不返回任何值，而是将处理后的文档存储在context.pdf_documents中
        """
        try:
            documents = []  # 存储所有 pdf 文件内容
            chunks = []  # 分割好的pdf chunks
            for file_path in context.pdf_files_path:
                with open(file_path, "rb") as pdf_file:
                    # 读入 pdf 文件

                    # 分割 pdf 文件

                    pass
            context.pdf_chunks = chunks
            # 生成文档
            # context.all_documents = documents
        except Exception as e:
            print(f"处理PDF文件时发生错误: {str(e)}")
            return []
    
    def split_pdf(self, pdf_document: list):
        # 读入pdf文件列表
        # 分割pdf文件
        # 返回分割好的pdf chunks
        pass

doc_split = DocSplit()

if __name__ == "__main__":
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert("/home/zhuhaiyang/RAG-main/初赛训练数据集.pdf")
    print(result.text_content)