from query_processor import QueryProcessor
from query_context import QueryContext
from llm_response import llm_response
from time import time

class GetAnswer(QueryProcessor):
    def __init__(self):
        super().__init__()
        self.name = "GetAnswer"

    def process(self, context: QueryContext) -> None:
        """
            List[str] -> str:

            处理QueryContext，根据用户问题和final_chunks生成最终答案。
        """
        try:
            start_time = time()  # 记录开始时间
            # 拼接final_chunks作为上下文
            context.context_text = "\n".join(context.final_chunks)  # 将文档内容保存到context中
            # 原始用户问题
            origin_query = context.user_origin_query
            # 构建提示语
            user_query = f"根据以下内容回答用户的问题。\n\n用户问题：{origin_query}\n\n相关文档：\n{context.context_text}\n\n答案："
            # 调用LLM生成最终答案
            context.final_answer = llm_response.process(user_query)
            end_time = time()  # 记录结束时间
            print(f"生成答案API调用耗时: {end_time - start_time} 秒")

        except Exception as e:
            print(f"生成答案失败: {e}")
            raise

get_answer = GetAnswer()

# 测试代码
if __name__ == "__main__":
    # 创建测试用的QueryContext对象
    context = QueryContext("这是什么东西?")
    context.rerank_chunks = [
        {"text": "这是一个测试文档，用于测试答案生成功能。"}
    ]
    
    try:
        get_answer.process(context)
        print("\n生成的答案:")
        print(context.final_answer)
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
