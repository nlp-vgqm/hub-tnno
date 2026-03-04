# -*- coding: utf-8 -*-

"""
Function Calling核心逻辑
处理与大模型的交互，解析function call请求并执行
"""

import json
import logging
from typing import Dict, Any, List, Optional
from functions import get_function_schemas, call_function

logger = logging.getLogger(__name__)


class FunctionCaller:
    """Function Calling处理器"""
    
    def __init__(self, config):
        self.config = config
        self.function_schemas = get_function_schemas()
        self.conversation_history = []
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """获取函数schema列表"""
        return self.function_schemas
    
    def parse_function_call(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析模型返回的function call
        
        Args:
            response: 模型响应字典
        
        Returns:
            解析后的function call信息，如果没有则返回None
        """
        # OpenAI格式：检查choices[0].message.tool_calls
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {})
            
            # 检查tool_calls（OpenAI新格式）
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                return {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": json.loads(tool_call["function"]["arguments"])
                    }
                }
            
            # 检查function_call（OpenAI旧格式）
            if "function_call" in message:
                func_call = message["function_call"]
                return {
                    "type": "function",
                    "function": {
                        "name": func_call["name"],
                        "arguments": json.loads(func_call["arguments"])
                    }
                }
        
        # 检查直接包含function_call的情况
        if "function_call" in response:
            func_call = response["function_call"]
            return {
                "type": "function",
                "function": {
                    "name": func_call["name"],
                    "arguments": json.loads(func_call["arguments"]) if isinstance(func_call["arguments"], str) else func_call["arguments"]
                }
            }
        
        return None
    
    def execute_function_call(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行function call
        
        Args:
            function_call: function call信息
        
        Returns:
            函数执行结果
        """
        func_name = function_call["function"]["name"]
        func_args = function_call["function"]["arguments"]
        
        logger.info(f"调用函数：{func_name}，参数：{func_args}")
        
        result = call_function(func_name, func_args)
        
        logger.info(f"函数执行结果：{result}")
        
        return result
    
    def format_function_result(self, function_call: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化函数执行结果，用于返回给模型
        
        Args:
            function_call: 原始function call
            result: 函数执行结果
        
        Returns:
            格式化后的结果
        """
        func_name = function_call["function"]["name"]
        
        # OpenAI格式
        formatted = {
            "role": "tool",
            "name": func_name,
            "content": json.dumps(result, ensure_ascii=False)
        }
        
        # 如果有tool_call_id，添加它
        if "id" in function_call:
            formatted["tool_call_id"] = function_call["id"]
        
        return formatted
    
    def add_to_history(self, role: str, content: Any, name: Optional[str] = None):
        """添加消息到对话历史"""
        message = {"role": role}
        
        if name:
            message["name"] = name
        
        if isinstance(content, dict):
            message.update(content)
        else:
            message["content"] = content
        
        self.conversation_history.append(message)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
