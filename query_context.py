
class QueryContext:
    """
    QueryContext类用于存储和传递查询处理过程中的上下文信息。
    包含原始查询、文档块及其在不同处理阶段的状态。
    """

    def __init__(self, user_original_query: str = ""):
        self.user_original_query = user_original_query  # 用户的原始查询 (str)
        self.rewritten_query = ""  # 重写后的用户查询 (str)
        self.query_keywords = []  # 重写后的用户查询的关键字列表 (list[str])
        self.user_query = user_original_query  # 处理过程中生成的中间问题 (str)
        self.pdf_files_path = []  # 存储读入的 pdf 文件路径 (list[str])
        self.pdf_chunks = []  # 存储分割好的 pdf chunks (list[str])
        self.pdf_documents = []  # 存储分割好的 pdf 文档 (list[document])
        self.pdf_chunk_sizes = []  # 存储 pdf 文件的分块大小 (list[int])
        self.all_texts = []  # 存储需要被 embedding 的文本 (list[str])
        self.all_texts_vectors = []  # 存储被 embedding 的文档的向量 (list[float])

        self.origin_chunks = []  # 原始检索到的文档块列表 (list[str])
        self.filter_chunks = []  # 经过过滤后的文档块列表 (list[str])
        self.rerank_chunks = []  # 经过重排序后的文档块列表 (list[str])
        self.final_chunks = []  # 最终的文档块列表 (list[str])
        self.final_questions = []  # 最终的问题列表 (list[str])
        self.generated_questions = []  # 基于文档内容生成的相关问题列表 (list[str])
        self.context_text = ""  # 用于生成答案的上下文文本 (str)
        self.llm_response = ""  # LLM模型的响应文本 (str)
        self.final_answer = ""  # 最终生成的答案文本 (str)

        self.embedding_model_name = None  # 用于文本向量化的模型名称 (str)
        self.llm_model_name = None  # 用于生成回答的大语言模型名称 (str)
        self.rerank_model_name = None  # 用于文档重排序的模型名称 (str)
        self.device = None  # 模型运行的设备类型,如'cpu'或'cuda' (str)

        self.csv_files_path = []  # 存储读入的 csv文件路径 (list[str])
        self.csv_chunks = []  # 存储分割好的 csv chunks (list[str])
        self.txt_files_path = []  # 存储读入的 txt 文件路径 (list[str])
        self.txt_chunks = []  # 存储分割好的 txt chunks (list[str])
        self.csv_chunk_sizes = []  # 存储 csv 文件的分块大小 (list[int])
        self.txt_chunk_sizes = []  # 存储 txt 文件的分块大小 (list[int])

        # 后续需要删除的参数
        self.index_questions = None
        self.index_documents = None 
        self.generated_questions = []
        self.question_to_doc_mapping = {}
        self.documents = []
        self.vectorizer_params = {}
        self.bm25_model = None
        

