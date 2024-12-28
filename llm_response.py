from query_processor import QueryProcessor
from query_context import QueryContext
import json
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
class LLMResponse(QueryProcessor):
    """
        该类负责调用OpenAI API获取LLM的响应结果。
    """
    
    def __init__(self, model_name="models/Qwen2.5-7B-Instruct", device="cpu"):
        super().__init__()
        self.name = "LLMResponse"
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            # device_map="auto"   # 运行在cuda环境则使用该参数
            device_map=self.device
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
        str -> str:

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
    

llm_response = LLMResponse()

# 测试代码
if __name__ == "__main__":
    import time
    
    start_time = time.time()
    response = llm_response.process("这是什么东西。")
    end_time = time.time()
    
    print(f"Response: {response}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")