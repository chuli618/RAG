from query_processor import QueryProcessor
from query_context import QueryContext
import json
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMResponse(QueryProcessor):
    """
    该类负责调用OpenAI API获取LLM的响应结果。
    """
    
    def __init__(self, model_name="models/Qwen2.5-7B-Instruct", device="cpu"):
        super().__init__()
        self.name = "LLMResponse"
        # self.openai_api_key = openai_api_key
        # self.openai_base_url = openai_base_url
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
        )
        self.max_token = 4096 # 最大输入token数量，暂未使用
    
    def process(self, user_query, system_prompt="You are a helpful assistant.") -> str:
        """
        调用大语言模型处理用户输入的查询，生成回复。

        Args:
            user_query (str): 用户输入的查询文本
            system_prompt (str): 系统提示词，用于指导模型的行为，默认为"You are a helpful assistant."
            
        Returns:
            str: 模型生成的回复文本
        """
        # input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        # if len(input_ids) > self.max_token:
        #     context = self.tokenizer.decode(input_ids[:self.max_token-1])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512 # 最多生成512个token
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
        # payload = json.dumps({
        #     # "model": "gpt-3.5-turbo",
        #     "model": "gpt-4o",
        #     'messages':[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_query}
        #     ],
        #     "safe_mode": False,
        #     "max_tokens": 100,
        #     "temperature": 0.7,
        #     "n": 1,
        # })
        
        # headers = {
        #     'Authorization': 'Bearer ' + self.openai_api_key,
        #     'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 
        #     'Content-Type': 'application/json'
        # }
        
        # response = requests.request("POST", self.openai_base_url, headers=headers, data=payload)
        # response_data = response.json()
        # return response_data['choices'][0]['message']['content']
    


# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_base_url = os.getenv("OPENAI_BASE_URL")
# llm_response = LLMResponse(openai_api_key, openai_base_url)
llm_response = LLMResponse()

# 测试代码
if __name__ == "__main__":
    response = llm_response.process("这是什么东西。")
    print(response)