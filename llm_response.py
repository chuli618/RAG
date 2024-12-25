from query_processor import QueryProcessor
from query_context import QueryContext
import json
import requests
import os

class LLMResponse(QueryProcessor):
    """
    该类负责调用OpenAI API获取LLM的响应结果。
    """
    
    def __init__(self, openai_api_key, openai_base_url):
        super().__init__()
        self.name = "LLMResponse"
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
    
    def process(self, user_query) -> str:
        """
        处理QueryContext，调用OpenAI API获取响应。
        
        Args:
            context: QueryContext对象，包含用户查询
        """
        payload = json.dumps({
            "model": "gpt-3.5-turbo",
            'messages':[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_query}
            ],
            "safe_mode": False,
            "max_tokens": 100,
            "temperature": 0.7,
            "n": 1,
        })
        
        headers = {
            'Authorization': 'Bearer ' + self.openai_api_key,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", self.openai_base_url, headers=headers, data=payload)
        response_data = response.json()
        return response_data['choices'][0]['message']['content']


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
llm_response = LLMResponse(openai_api_key, openai_base_url)

# 测试代码
if __name__ == "__main__":
    response = llm_response.process("这是什么东西。")
    print(response)