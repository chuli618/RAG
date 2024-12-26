from query_processor import QueryProcessor
from query_context import QueryContext
import faiss
import numpy as np
from time import time
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer


class DocEmbedding(QueryProcessor):
    def __init__(self, model_name="bge-base-zh-v1.5"):
        super().__init__()
        self.name = "DocEmbedding"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def process(self, context: QueryContext) -> None:
        """
        处理QueryContext，为文档和问题生成向量嵌入。
        
        Args:
            context: QueryContext对象，包含原始文档块和生成的问题
        """
        documents = context.all_documents
        if not documents:
            return np.empty((0, 768), dtype='float32')
        vector_model = self.get_vector_model(context)
        vectors = vector_model.encode(
            documents,
            # convert_to_numpy=True, # 直接输出numpy数组
        ) 
             
        print(f"向量维度: {vectors.shape}")  # 添加维度信息日志
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)  # 修复单个向量的维度问题
        else:
            raise TypeError("向量化结果不是 numpy.ndarray")
        context.all_documents_vectors = vectors.astype('float32')  # 确保向量类型为 float32，FAISS 更兼容
    
    def get_vector_model(self, context: QueryContext):
        """
        根据指定的模型类型初始化并返回向量化模型。
        
        参数:
            model_type (str): 模型类型，'local' 或 'azure_openai'。
            model_name (str): 本地模型名称。
            device (str): 设备类型，'cpu' 或 'cuda'。
        
        返回:
            object: 初始化后的模型对象。
        """
        model_name = context.model_name
        device = context.device
        model_name = "models/" + model_name
        return SentenceTransformer(model_name, device=device)
    
    def vectorize_chunks(self, chunks, model_name='bge-base-zh-v1.5', device='cpu'):
        if not chunks:
            return np.empty((0, 768), dtype='float32')
        model = self.get_vector_model(model_name, device)
        vectors = model.encode(chunks)
        print(f"向量维度: {vectors.shape}")  # 添加维度信息日志
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)  # 修复单个向量的维度问题
            else:
                raise TypeError("向量化结果不是 numpy.ndarray")
        return vectors.astype('float32')  # 确保向量类型为 float32，FAISS 更兼容

    def build_vector_db(self, question_vectors: np.ndarray, document_vectors: np.ndarray):
        """
        构建问题和文档的向量数据库。

        参数:
            question_vectors (numpy.ndarray): 问题的向量表示。
            document_vectors (numpy.ndarray): 文档的向量表示。

        返回:
            dict: 包含问题索引和文档索引的字典。
        """
        if not isinstance(question_vectors, np.ndarray) or not isinstance(document_vectors, np.ndarray):
            raise TypeError("所有向量必须是 numpy.ndarray")
        
        indices = {}
        
        # 构建问题索引
        if question_vectors.size > 0:
            embedding_dim_q = question_vectors.shape[1]
            index_q = faiss.IndexFlatL2(embedding_dim_q)
            
            # 在添加向量之前打印维度信息
            print(f"问题向量维度: {question_vectors.shape}")
            print(f"FAISS 问题索引维度: {embedding_dim_q}")
            
            index_q.add(question_vectors)
            
            # 验证添加是否成功
            if index_q.ntotal != question_vectors.shape[0]:
                raise ValueError(f"FAISS 问题索引的向量数量 ({index_q.ntotal}) 与输入的向量数量 ({question_vectors.shape[0]}) 不匹配")
            
            indices['questions'] = index_q
        else:
            raise ValueError("没有可添加到FAISS索引的问题向量")
        
        # 构建文档索引
        if document_vectors.size > 0:
            embedding_dim_d = document_vectors.shape[1]
            index_d = faiss.IndexFlatL2(embedding_dim_d)
            
            # 在添加向量之前打印维度信息
            print(f"文档向量维度: {document_vectors.shape}")
            print(f"FAISS 文档索引维度: {embedding_dim_d}")
            
            index_d.add(document_vectors)
            
            # 验证添加是否成功
            if index_d.ntotal != document_vectors.shape[0]:
                raise ValueError("文档向量添加到 FAISS 索引失败")
            
            indices['documents'] = index_d
        else:
            raise ValueError("没有可添加到FAISS索引的文档向量")
        
        # 打印最终的索引信息
        print(f"构建完成 - 问题索引总数: {indices['questions'].ntotal}")
        print(f"构建完成 - 文档索引总数: {indices['documents'].ntotal}")
        
        return indices

doc_embedding = DocEmbedding()

# 测试代码
if __name__ == "__main__":
    # 创建测试用的QueryContext对象
    context = QueryContext()
    context.model_name = "bge-base-zh-v1.5"
    context.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 检查设备配置
    print(f"- 选择的设备: {context.device}")

    # 添加测试文档
    context.all_documents = [
        "这是第一个测试文档,主要讨论了向量化的基本概念。",
        "这是第二个测试文档,介绍了FAISS索引的构建方法。",
        "这是第三个测试文档,说明了向量相似度的计算原理。"
    ]
    
    try:
        # 执行向量化处理
        doc_embedding.process(context)
        
        # 打印处理结果
        print(f"- 向量: {context.all_documents_vectors}")
        print(f"- 向量维度: {context.all_documents_vectors.shape}")

    except Exception as e:
        print(f"向量化处理过程中发生错误: {str(e)}")
