
class QueryContext:
    """
    QueryContext类用于存储和传递查询处理过程中的上下文信息。
    包含原始查询、文档块及其在不同处理阶段的状态。
    """

    def __init__(self, user_original_query: str = ""):
        self.user_original_query = user_original_query  # 用户的原始查询文本
        self.user_query = user_original_query  # 处理过程中生成的中间问题
        self.csv_files_path = []  # 存储读入的csv文件路径
        self.csv_chunks = []  # 存储分割好的csv chunks
        self.pdf_files_path = []  # 存储读入的pdf文件路径
        self.pdf_chunks = []  # 存储分割好的pdf chunks
        self.txt_files_path = []  # 存储读入的txt文件路径
        self.txt_chunks = []  # 存储分割好的txt chunks
        self.csv_chunk_sizes = []  # 存储csv文件的分块大小
        self.pdf_chunk_sizes = []  # 存储pdf文件的分块大小
        self.txt_chunk_sizes = []  # 存储txt文件的分块大小
        self.all_documents = []  # 存储所有分割好的文档
        self.all_documents_vectors = []  # 存储所有分割好的文档的向量

        self.origin_chunks = []  # 原始检索到的文档块列表
        self.filter_chunks = []  # 经过过滤后的文档块列表
        self.rerank_chunks = []  # 经过重排序后的文档块列表
        self.final_chunks = []  # 最终的文档块列表
        self.final_questions = []  # 最终的问题列表
        self.generated_questions = []  # 基于文档内容生成的相关问题列表
        self.context_text = ""  # 用于生成答案的上下文文本
        self.llm_response = ""  # LLM模型的响应文本
        self.final_answer = ""  # 最终生成的答案文本

        self.model_name = None
        self.device = None

        # 后续需要删除的参数
        self.index_questions = None
        self.index_documents = None 
        self.generated_questions = []
        self.question_to_doc_mapping = {}
        self.documents = []
        self.vectorizer_params = {}
        self.bm25_model = None
        

