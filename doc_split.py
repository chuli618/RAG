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
            str -> List[str], List[document]

            传入pdf文件路径，分割pdf文件，返回分割好的pdf chunks

            参数:
                context (QueryContext): 包含处理所需参数的上下文对象
        """
        # 获取pdf文件路径
        pdf_files_path = context.pdf_files_path[0]
        try:
            chunks = []  # 存储分割好的pdf chunks
            documents = []  # 存储所有 pdf 文档
            # 读入pdf文件
            for file_path in pdf_files_path:
                with open(file_path, "rb") as pdf_file:
                    # 识别 pdf 文件

                    # 分割 pdf 文件

                    # 生成文档
                    
                    pass
            context.pdf_chunks = chunks
            context.pdf_documents = documents
        except Exception as e:
            print(f"处理PDF文件时发生错误: {str(e)}")
            return []

    # '''
    #     识别和分割 pdf文件chunk 的参考代码
    #     天池比赛第四名，用PyPDF2库识别pdf文件，使用滑动窗口进行 chunk 分割
    # '''
    # def extract_pdf_page_text(self, filepath):
    #     import tqdm
    #     import re
    #     max_len = 256   # chunk的最大长度
    #     overlap_len = 100  # chunk之间的重叠长度
    #     page_content  = []
    #     # 识别pdf文件，并做预处理
    #     with open(filepath, 'rb') as f:
    #         pdf_reader = PyPDF2.PdfReader(f)
    #         for page in tqdm.tqdm(pdf_reader.pages, desc='解析PDF文件...'):
    #             page_text = page.extract_text().strip()
    #             raw_text = [text.strip() for text in page_text.split('\n')]
    #             new_text = '\n'.join(raw_text)
    #             new_text = re.sub(r'\n\d{2,3}\s?', '\n', new_text)
    #             if len(new_text) > 10 and '..............' not in new_text:
    #                 page_content.append(new_text)

    #     cleaned_chunks = []
    #     i = 0
    #     # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
    #     all_str = ''.join(page_content)
    #     all_str = all_str.replace('\n', '')
    #     while i < len(all_str):
    #         cur_s = all_str[i:i+max_len]
    #         if len(cur_s) > 10:
    #             cleaned_chunks.append(cur_s)
    #         i += (max_len - overlap_len)

    #     return cleaned_chunks


doc_split = DocSplit()

if __name__ == "__main__":
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert("/home/zhuhaiyang/RAG-main/初赛训练数据集.pdf")
    print(result.text_content)