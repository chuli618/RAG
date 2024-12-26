from query_processor import QueryProcessor
from query_context import QueryContext
from llm_response import llm_response
import json

class QueryRewrite(QueryProcessor):
    """查询重写处理器，用于改写和优化用户查询"""
    
    def __init__(self):
        """初始化查询重写处理器"""
        super().__init__()
        self.name = "QueryRewrite"

    def process(self, context: QueryContext) -> None:
        """
        处理并重写用户的原始查询，优化查询效果。

        参数:
            context (QueryContext): 包含查询相关信息的上下文对象

        返回:
            None: 该方法不返回任何值，而是将处理后的查询存储在context中
        """
        try:
            if not context.user_original_query:
                print("用户查询为空")
                return
            user_query = context.user_original_query
            # 这里需要根据实际场景进行改写
            system_prompt = f"""
                你是一名代码问答智能助手。你需要扩展、拆解用户的问题，并提供用户可以搜索的关键词列表来帮助用户回答有关该仓库的问题。

                # 附加规则
                1. 阅读用户的问题，了解他们提出的关于代码仓库的问题，务必结合项目目录文件列表回答用户提问
                2. 在回答时，请先简单扩展分析用户的问题，再给出解答用户提问的步骤及其背后的思路、意义，最后给出搜索提问相关文件使用的关键词，以及用户应重点分析哪些文件。
                3. 关键词相关的额外规则：
                - 关键词可能提取于用户的原始提问，也有可能来自于扩展后的子查询
                - 将与问题最相关的关键词放在前面
                - 关键词应考虑变体，例如：对于"encode"，可能的变体包括"encoding"、"encoded"、"encoder"、"encoders"。考虑同义词和复数形式-
                - 不要包含过于通用的关键词
                4. 如果用户的提问与测试无关，你的回答不能涉及测试相关文件。
                5. 用英文回复，但对于你不确定如何翻译的业务词汇、技术词汇，请保留原样，不要翻译
                6. 请不要提及或要求用户提供更多信息。
                7. 不要试图直接回答用户的问题。
                8. 回复请严格遵循以下json格式：
                {{
                "description": "对用户问题的一句话分析",
                "subqueries":["step1", "step2"],
                "keywords":["keyword1", "keyword2"],
                }}

                # 示例1
                示例提问：找到这个项目对外暴露的所有http服务
                示例回答：
                {{
                "description": "用户想要了解这个项目提供的所有HTTP服务。",
                "subqueries":[
                "搜索通常处理HTTP请求的Java类。查找类似UserOperateCeontroller、ServiceGuideController、AlarmTestController或其他暗示web控制器的文件名。",
                "识别带有@RestController、@Controller、@RequestMapping、@GetMapping、@PostMapping、@PutMapping、@DeleteMapping等注解的方法。",
                "检查配置文件如application.properties、application.yml或特定环境的文件如prod.aci.yml中定义的服务器端口和上下文路径,这些都是HTTP服务的入口点。"
                ],
                "keywords":["Controller", "@RequestMapping", "@GetMapping", "@PostMapping"],
                }}

                # 示例2
                示例提问：这个项目是否引入了fastjson依赖？
                示例回答：
                {{
                "description": "用户想知道项目是否包含了fastjson依赖。",
                "subqueries":[
                "在`pom.xml`文件中搜索依赖项，查看是否列出了`fastjson`",
                "在Java文件中查找引用`fastjson`类的import语句"
                ],
                "keywords":["fastjson", "dependency", "import com.alibaba.fastjson"],
                }}

                # 按以上规则处理用户如下提问，并严格按照json格式回答：
            """

            response = llm_response.process(user_query, system_prompt)
            # 解析响应存储为json格式
            response_json = None
            try:
                response_json = json.loads(response)
            except Exception as e:
                print(f'llm的响应：{response}')

            if response_json:
                lines = []
                lines.append(response_json["description"])
                lines.extend(response_json["subqueries"])
                context.rewritten_query = ''.join(lines)
                context.query_keywords = response_json["keywords"]
            else:
                print("llm_response_json 解析失败")
        except Exception as e:
            print(f"查询重写过程中发生错误: {str(e)}")
            # 如果重写失败，则使用原始查询
            context.rewritten_query = context.user_query

query_rewrite = QueryRewrite()

if __name__ == "__main__":
    context = QueryContext()
    context.user_original_query = "这个项目有哪些核心模型？"
    query_rewrite.process(context)
    print(context.rewritten_query)
    print(context.query_keywords)