from query_processor import QueryProcessor
from query_context import QueryContext
from doc_embedding import doc_embedding
from langchain.vectorstores import FAISS
import jieba
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS

class VectorDB(QueryProcessor):
    """
        该类负责从文档库中检索与用户查询相关的文档。
        继承自QueryProcessor基类。
    """
    
    def __init__(self, embedding_model_name="models/bge-large-zh-v1.5", lan='zh'):
        super().__init__()
        self.name = "VectorDB"  
        self.embedding_model_name = embedding_model_name
        self.lan = lan
        self.bm25_model = None
        self.db = None

    def process(self, context: QueryContext) -> None:
        """
            List[str] -> bm25检索模型 和 faiss向量数据库

            使用 jieba 分词构建 bm25检索模型
            使用 langchain 的 FAISS 构建向量数据库
        """
        # 读取切割好的 chunks
        self.pdf_chunks = context.pdf_chunks # List[str]
        # 将chunks转换为Document对象, 暂时只保存文本，元数据为空
        self.pdf_documents = [Document(page_content=t) for t in self.pdf_chunks]
        # 将构建好的文档存储在context中
        context.pdf_documents = self.pdf_documents
        # 对文档进行分词，如果语言是中文，则使用jieba进行分词，否则使用空格分词
        if self.lan=='zh':
            jieba_documents = [jieba.lcut(doc) for doc in self.pdf_chunks]
        else:
            jieba_documents = [doc.split() for doc in self.pdf_chunks]

        # 构建 bm25 检索模型
        self.bm25_model = BM25Okapi(jieba_documents)
        # 构建 faiss 向量数据库，使用文档和向量模型   
        self.vector_db = FAISS.from_documents(self.pdf_documents, doc_embedding)

        # 将构建好的 bm25模型和向量数据库存储在 context 中
        context.bm25_model = self.bm25_model
        context.vector_db = self.vector_db
    
vector_db = VectorDB()