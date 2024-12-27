from query_processor import QueryProcessor
from query_context import QueryContext
import faiss
import numpy as np
from time import time
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class DocEmbedding(QueryProcessor):
    def __init__(self, model_name="models/bge-large-zh-v1.5", device="cuda"):
        super().__init__()
        self.name = "DocEmbedding" # embedding维度：1024
        self.device = device
        self.model_name = model_name
        self.batch_size=64
        self.max_len=512
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if 'bge' in model_name:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        
        print("successful load embedding model")


    def process(self, context: QueryContext) -> None:
        """
        List[str] -> List[List[float]]

        处理QueryContext对象，将文本内容转换为向量嵌入表示。
        
        Args:
            context (QueryContext): 包含待处理文档的上下文对象。context.all_texts中存储了需要向量化的文本列表。
            
        Returns:
            List[List[float]]: 返回文本的向量嵌入表示列表。每个文本对应一个向量，向量维度由模型决定。
            同时会将向量存储在context.all_texts_vectors中。
        """
        texts = context.all_texts  # texts: List[str]

        num_texts = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        text_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.model_name:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                text_embeddings.extend(batch_embeddings.tolist())

        context.all_texts_vectors = text_embeddings # List[List[float]]
    

    # def build_vector_db(self, question_vectors: np.ndarray, document_vectors: np.ndarray):
    #     """
    #     构建问题和文档的向量数据库。

    #     参数:
    #         question_vectors (numpy.ndarray): 问题的向量表示。
    #         document_vectors (numpy.ndarray): 文档的向量表示。

    #     返回:
    #         dict: 包含问题索引和文档索引的字典。
    #     """
    #     if not isinstance(question_vectors, np.ndarray) or not isinstance(document_vectors, np.ndarray):
    #         raise TypeError("所有向量必须是 numpy.ndarray")
        
    #     indices = {}
        
    #     # 构建问题索引
    #     if question_vectors.size > 0:
    #         embedding_dim_q = question_vectors.shape[1]
    #         index_q = faiss.IndexFlatL2(embedding_dim_q)
            
    #         # 在添加向量之前打印维度信息
    #         print(f"问题向量维度: {question_vectors.shape}")
    #         print(f"FAISS 问题索引维度: {embedding_dim_q}")
            
    #         index_q.add(question_vectors)
            
    #         # 验证添加是否成功
    #         if index_q.ntotal != question_vectors.shape[0]:
    #             raise ValueError(f"FAISS 问题索引的向量数量 ({index_q.ntotal}) 与输入的向量数量 ({question_vectors.shape[0]}) 不匹配")
            
    #         indices['questions'] = index_q
    #     else:
    #         raise ValueError("没有可添加到FAISS索引的问题向量")
        
    #     # 构建文档索引
    #     if document_vectors.size > 0:
    #         embedding_dim_d = document_vectors.shape[1]
    #         index_d = faiss.IndexFlatL2(embedding_dim_d)
            
    #         # 在添加向量之前打印维度信息
    #         print(f"文档向量维度: {document_vectors.shape}")
    #         print(f"FAISS 文档索引维度: {embedding_dim_d}")
            
    #         index_d.add(document_vectors)
            
    #         # 验证添加是否成功
    #         if index_d.ntotal != document_vectors.shape[0]:
    #             raise ValueError("文档向量添加到 FAISS 索引失败")
            
    #         indices['documents'] = index_d
    #     else:
    #         raise ValueError("没有可添加到FAISS索引的文档向量")
        
    #     # 打印最终的索引信息
    #     print(f"构建完成 - 问题索引总数: {indices['questions'].ntotal}")
    #     print(f"构建完成 - 文档索引总数: {indices['documents'].ntotal}")
        
    #     return indices

doc_embedding = DocEmbedding()

# 测试代码
if __name__ == "__main__":
    # 创建测试用的QueryContext对象
    context = QueryContext("这是一个测试问题")
    
    # 添加一些测试文档
    context.all_texts = [
        "这是第一个测试文档。",
        "这是第二个测试文档。",
        "这是第三个测试文档。",
    ]
    
    try:
        # 调用process函数进行文档向量化
        start_time = time()
        doc_embedding.process(context)
        end_time = time()
        print(f"文档向量化耗时: {end_time - start_time} 秒")
        
        # 验证结果
        print("测试结果:")

        text_embeddings = context.all_texts_vectors
        # 计算行数和列数
        num_rows = len(text_embeddings)
        num_cols = len(text_embeddings[0]) if num_rows > 0 else 0

        # 打印形状
        print(f"文档向量的形状: ({num_rows}, {num_cols})")
  
    except Exception as e:
        print(f"测试过程中发生错误: {e}")