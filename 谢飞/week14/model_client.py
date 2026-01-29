# -*- coding: utf-8 -*-

"""
大模型客户端
支持OpenAI API和本地模型
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ModelClient:
    """大模型客户端基类"""
    
    def __init__(self, config):
        self.config = config
    
    def chat(self, messages: List[Dict[str, Any]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            functions: 函数schema列表
        
        Returns:
            模型响应
        """
        raise NotImplementedError


class OpenAIClient(ModelClient):
    """OpenAI API客户端"""
    
    def __init__(self, config):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config["openai_api_key"],
                base_url=config.get("openai_base_url")
            )
            self.model = config["openai_model"]
        except ImportError:
            raise ImportError("请安装openai库：pip install openai")
        except Exception as e:
            raise ValueError(f"OpenAI客户端初始化失败：{e}")
    
    def chat(self, messages: List[Dict[str, Any]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """发送聊天请求到OpenAI"""
        try:
            # 准备请求参数
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 1000),
            }
            
            # 如果有函数，添加function calling相关参数
            if functions and self.config.get("enable_function_calling", True):
                params["tools"] = [{"type": "function", "function": func} for func in functions]
                params["tool_choice"] = self.config.get("function_call_auto", "auto")
            
            # 发送请求
            response = self.client.chat.completions.create(**params)
            
            # 转换为字典格式
            result = {
                "choices": [{
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content,
                    }
                }]
            }
            
            # 处理tool_calls
            if response.choices[0].message.tool_calls:
                result["choices"][0]["message"]["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            
            # 处理function_call（旧格式）
            if hasattr(response.choices[0].message, "function_call") and response.choices[0].message.function_call:
                result["choices"][0]["message"]["function_call"] = {
                    "name": response.choices[0].message.function_call.name,
                    "arguments": response.choices[0].message.function_call.arguments
                }
            
            return result
        
        except Exception as e:
            logger.error(f"OpenAI API调用失败：{e}")
            raise


class LocalModelClient(ModelClient):
    """本地模型客户端（使用transformers）"""
    
    def __init__(self, config):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_path = config["local_model_path"]
            if model_path is None:
                raise ValueError("local_model_path 未配置")
            
            device = config.get("local_device", "cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"加载本地模型：{model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.device = device
            logger.info("本地模型加载完成")
        
        except ImportError:
            raise ImportError("请安装transformers库：pip install transformers")
        except Exception as e:
            raise ValueError(f"本地模型加载失败：{e}")
    
    def chat(self, messages: List[Dict[str, Any]], functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        本地模型聊天（简化实现）
        注意：本地模型可能不支持function calling，这里只是示例
        """
        # 构建提示词
        prompt = self._build_prompt(messages, functions)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get("max_tokens", 1000),
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }]
        }
    
    def _build_prompt(self, messages: List[Dict[str, Any]], functions: Optional[List[Dict[str, Any]]] = None) -> str:
        """构建提示词"""
        prompt = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"系统：{content}\n\n"
            elif role == "user":
                prompt += f"用户：{content}\n\n"
            elif role == "assistant":
                prompt += f"助手：{content}\n\n"
        
        prompt += "助手："
        return prompt


def create_model_client(config: Dict[str, Any]) -> ModelClient:
    """
    创建模型客户端
    
    Args:
        config: 配置字典
    
    Returns:
        模型客户端实例
    """
    model_type = config.get("model_type", "openai")
    
    if model_type == "openai":
        return OpenAIClient(config)
    elif model_type == "local":
        return LocalModelClient(config)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
