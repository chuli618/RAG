from query_processor import QueryProcessor
from query_context import QueryContext
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class DocRerank(QueryProcessor):
    def __init__(self, model_name="models/bge-reranker-large", device="cuda"):
        super().__init__()
        self.name = "DocRerank"
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)\
            .half().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.top_k = 5
        print("successful load rerank model")


    def process(self, context: QueryContext) -> None:
        """
        List[str] -> top_k List[str]:

        对检索到的原始文档进行重排序，根据与query的相关性得分重新排序，
        并更新context的filter_chunks、rerank_chunks、final_chunks

        Args:
            context (QueryContext): 包含用户查询和原始文档列表
                - context.user_original_query: 用户查询
                - context.original_chunks: 原始文档列表
        """
        user_query = context.user_original_query  # str 
        original_chunks = context.original_chunks   # List[str] 原始文档列表

        chunks_ = []
        # 保证数据类型可哈希，可被set处理
        for item in original_chunks:
            if isinstance(item, str):
                chunks_.append(item)
        # 去重过滤
        filter_chunks = list(set(chunks_))
        context.filter_chunks = filter_chunks
        # rerank
        pairs = []
        # 构建 query-chunk 对
        for c in filter_chunks:
            pairs.append([user_query, c])
        # 计算相关性得分
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)\
                .to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
        filter_chunks = [(filter_chunks[i], scores[i]) for i in range(len(filter_chunks))]
        # 根据相关性得分排序
        rerank_chunks = sorted(filter_chunks, key = lambda x: x[1], reverse = True)
        context.rerank_chunks = rerank_chunks
        # 获取排序后的文档
        chunks_ = []
        for item in rerank_chunks:
            chunks_.append(item[0])
        # 获取 rerank 后的 top_k 个文档
        context.final_chunks = chunks_[: self.top_k]

if __name__ == "__main__":
    context = QueryContext()
    context.user_original_query = "什么是AI？"
    context.original_chunks = [
        "人工智能(AI)是计算机科学的一个分支,它试图理解智能的本质,并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "AI包括机器学习、深度学习、自然语言处理等多个领域,已广泛应用于图像识别、语音识别、自动驾驶等场景。",
        "AI是一种技术",
        "尽管AI取得了巨大进展,但在通用人工智能、意识和情感等方面仍面临巨大挑战,需要科研人员持续探索和突破。",
        "AI是一种很新很强大的技术。"
    ]
    doc_rerank = DocRerank()
    doc_rerank.process(context)
    print(context.final_chunks)