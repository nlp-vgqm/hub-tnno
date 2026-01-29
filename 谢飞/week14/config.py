# -*- coding: utf-8 -*-

"""
配置文件
基于大模型的Function Calling任务配置
"""

import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Config = {
    # 大模型配置
    # 支持OpenAI API或本地模型
    "model_type": "openai",  # "openai" 或 "local"
    
    # OpenAI配置（如果使用OpenAI API）
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),  # 从环境变量读取
    "openai_model": "gpt-3.5-turbo",  # 或 "gpt-4", "gpt-4-turbo"
    "openai_base_url": None,  # 如果使用代理，设置base_url
    
    # 本地模型配置（如果使用本地模型，需要安装transformers）
    "local_model_path": None,  # 例如: "THUDM/chatglm3-6b"
    "local_device": "cuda",  # "cuda" 或 "cpu"
    
    # 对话配置
    "temperature": 0.7,  # 生成温度
    "max_tokens": 1000,  # 最大生成token数
    "system_prompt": """你是一个智能助手，可以帮助用户完成各种任务。
你可以调用以下函数来帮助用户：
- calculator: 执行数学计算
- get_current_time: 获取当前时间
- weather_query: 查询天气信息
- text_processing: 处理文本（翻译、摘要等）
- database_query: 查询数据库信息

当用户需要这些功能时，请主动调用相应的函数。""",
    
    # Function Calling配置
    "enable_function_calling": True,  # 是否启用function calling
    "function_call_auto": "auto",  # "auto"（自动）或 "none"（禁用）或函数名（强制调用）
}
