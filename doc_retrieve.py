from query_processor import QueryProcessor
from query_context import QueryContext
from doc_embedding import doc_embedding

class DocRetrieve(QueryProcessor):
    """
    该类负责从文档库中检索与用户查询相关的文档。
    继承自QueryProcessor基类。
    """
    
    def __init__(self):
        super().__init__()
        self.name = "DocRetrieve"

    def process(self, context: QueryContext) -> None:
        try:
            user_query = context.user_original_query
            index_questions = context.index_questions
            index_documents = context.index_documents
            questions = context.questions
            question_to_doc_mapping = context.question_to_doc_mapping
            documents = context.documents
            vectorizer_params = context.vectorizer_params
            bm25_model = context.bm25_model
            # 向量化用户查询
            context.all_documents = [user_query]
            doc_embedding.process(context)
            query_vector = context.all_documents_vectors

            # 确保向量维度匹配
            if index_questions is not None:
                expected_dim = index_questions.d  # 获取 FAISS 索引的维度
                if query_vector.shape[1] != expected_dim:
                    raise AssertionError(f"查询向量的维度 {query_vector.shape[1]} 不匹配 FAISS 索引的维度 {expected_dim}")
            
            k = 3  # 设置检索的 top 数量

            # 执行向量搜索
            if index_questions is not None and index_documents is not None:
                # 在问题索引中搜索最相似的 k 个向量
                D_q, I_q = index_questions.search(query_vector, k)
                
                # 添加日志，显示检索到的相关问题数量
                num_related_questions = len([idx for idx in I_q[0] if idx >= 0])
                print(f"L2 向量检索找到的相关问题数量: {num_related_questions}")

                # 处理问题索引的结果
                for idx, distance in zip(I_q[0], D_q[0]):
                    if 0 <= idx < len(questions):
                        question = questions[idx]
                        doc_id = question_to_doc_mapping.get(question)
                        if doc_id and not doc_id.startswith("processed_"):  # 排除处理文件
                            context.final_questions.append({
                                "question": question,
                                "doc_id": doc_id,
                                "distance": float(distance)  # 确保距离是 Python 原生类型
                            })
                    else:
                        print(f"问题索引超出范围: {idx}")

                # 在文档索引中搜索最相似的 k 个向量
                D_d, I_d = index_documents.search(query_vector, k)
                
                # 添加日志，显示检索到的相关文档数量
                num_related_documents = len([idx for idx in I_d[0] if idx >= 0])
                print(f"L2 向量检索找到的相关文档数量: {num_related_documents}")

                # 处理文档索引的结果
                for idx, distance in zip(I_d[0], D_d[0]):
                    if 0 <= idx < len(documents):
                        doc = documents[idx]
                        if not doc["id"].startswith("processed_"):  # 排除处理文件
                            context.final_chunks.append({
                                "doc_id": doc["id"],
                                "text": doc["text"],
                                "distance": float(distance)  # 确保距离是 Python 原生类型
                            })
                    else:
                        print(f"文档索引超出范围: {idx}")
            else:
                print("未选择向量搜索，跳过向量检索。")

            # 仅当 bm25_model 不为 None 时执行 BM25 检索
            if bm25_model is not None:
                print("开始BM25检索...")
                tokenized_query = user_query.split()
                bm25_scores = bm25_model.get_scores(tokenized_query)
                top_n = 3
                bm25_top_n = bm25_scores.argsort()[-top_n:][::-1]
                print(f"BM25检索得到的文档索引: {bm25_top_n}")

                # 收集 BM25 结果
                bm25_results = []
                for idx in bm25_top_n:
                    if 0 <= idx < len(documents):
                        doc = documents[idx]
                        bm25_results.append({
                            "doc_id": doc["id"],
                            "text": doc["text"],
                            "bm25_score": float(bm25_scores[idx])
                        })
                        print(f"BM25找到相关文档: {doc['id']} (得分: {bm25_scores[idx]})")
                    else:
                        print(f"BM25文档索引超出范围: {idx}")
            else:
                print("未选择BM25检索或 BM25 模型未加载，跳过BM25检索。")
                bm25_results = []

            # 合并结果并去重
            all_documents = context.final_chunks + bm25_results

            # 根据 doc_id 去重并合并分数
            unique_docs = {}
            for doc in all_documents:
                doc_id = doc["doc_id"]
                if doc_id in unique_docs:
                    existing_doc = unique_docs[doc_id]
                    if "distance" in doc:
                        existing_doc["distance"] = min(existing_doc.get("distance", float('inf')), doc["distance"])
                    if "bm25_score" in doc:
                        existing_doc["bm25_score"] = max(existing_doc.get("bm25_score", 0), doc["bm25_score"])
                else:
                    unique_docs[doc_id] = doc

            # 排序文档
            def doc_sort_key(doc):
                # BM25 得分越高越好，距离越小越好
                bm25_score = doc.get("bm25_score", 0)
                distance = doc.get("distance", float('inf'))
                return (-bm25_score, distance)

            final_documents = sorted(unique_docs.values(), key=doc_sort_key)
            
            # 限制为 top 9 个文档
            final_documents = final_documents[:9]

            # 更新结果
            # results["related_documents"] = final_documents

            # # 返回结果
            # return results

            context.final_chunks = final_documents

        except AssertionError as ae:
            print(f"AssertionError: {ae}")
            raise
        except Exception as e:
            print(f"检索过程中发生错误: {str(e)}")
            # 不抛出异常，避免程序崩溃
            #raise

doc_retrieve = DocRetrieve()