from query_processor import QueryProcessor
from query_context import QueryContext
from llm_response import llm_response
from time import time

class QueryGenerate(QueryProcessor):
    def __init__(self):
        super().__init__()
        self.name = "QueryGenerate"

    def process(self, context: QueryContext) -> None:
        """
        处理QueryContext，生成相关问题。
        
        Args:
            context: QueryContext对象，包含文档块
        """
        questions = []  # 初始化问题列表
        question_to_doc_mapping = {}
        for doc in context.all_documents:
            try:
                start_time = time()  # 记录开始时间
                user_query = f"根据以下内容生成3个相关问题：\n{doc['text']}"
                context.llm_response = llm_response.process(user_query)
                end_time = time()  # 记录结束时间
                print(f"API 调用耗时: {end_time - start_time} 秒")  # 打印 API 调用时间

                if context.llm_response:
                    # 假设生成的问题以换行符分隔
                    doc_questions = context.llm_response.strip().split("\n")
                    # 清理和过滤生成的问题
                    cleaned_questions = [q.strip() for q in doc_questions if q.strip()]
                    questions.extend(cleaned_questions)
                    question_to_doc_mapping.update({q: doc["id"] for q in cleaned_questions})
                    print(f"生成了 {len(cleaned_questions)} 个问题。")  # 添加调试信息
                else:
                    raise ValueError("API 响应中没有找到 'choices' 字段")
            except Exception as e:
                print(f"生成问题失败: {e}")  # 添加调试信息
                raise
        context.generated_questions = questions  # 将生成的问题存储在context中
        context.question_to_doc_mapping = question_to_doc_mapping  # 将问题到文档的映射存储在context中
    
    def generate_query(self, documents) -> tuple:
        """
        处理documents，生成相关问题。
        
        Args:
            documents: list, 包含多个文档块的列表，每个文档块包含text和id字段
        """
        questions = []  # 初始化问题列表
        question_to_doc_mapping = {}

        for doc in documents:
            user_query = f"根据以下内容生成3个相关问题：\n{doc['text']}"
            response = llm_response.process(user_query)
            if response:
                # 假设生成的问题以换行符分隔
                doc_questions = response.strip().split("\n")
                # 清理和过滤生成的问题
                cleaned_questions = [q.strip() for q in doc_questions if q.strip()]
                questions.extend(cleaned_questions)
                question_to_doc_mapping.update({q: doc["id"] for q in cleaned_questions})
                print(f"生成了 {len(cleaned_questions)} 个问题。")  # 添加调试信息
            else:
                raise ValueError("API 响应中没有找到 'choices' 字段")
        return questions, question_to_doc_mapping
        
query_generate = QueryGenerate()


