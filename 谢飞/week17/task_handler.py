# -*- coding: utf-8 -*-

"""
任务处理器
处理任务型对话的核心业务逻辑
"""

import logging
from typing import Dict, Any, Optional
from dialogue_history import DialogueHistory
from function_caller import FunctionCaller
from model_client import ModelClient
from replay_handler import ReplayHandler

logger = logging.getLogger(__name__)


class TaskHandler:
    """任务型对话处理器"""
    
    def __init__(self, config, model_client: ModelClient, dialogue_history: DialogueHistory, 
                 function_caller: FunctionCaller, replay_handler: ReplayHandler):
        self.config = config
        self.model_client = model_client
        self.dialogue_history = dialogue_history
        self.function_caller = function_caller
        self.replay_handler = replay_handler
    
    def process_user_input(self, user_input: str, max_iterations: int = 5) -> str:
        """
        处理用户输入，支持任务执行和重听功能
        
        Args:
            user_input: 用户输入
            max_iterations: 最大迭代次数（防止无限循环）
        
        Returns:
            最终回复
        """
        # 首先检查是否是重听请求
        replay_response = self.replay_handler.handle_replay_request(user_input)
        if replay_response:
            logger.info("检测到重听请求，直接返回之前的回复")
            # 将重听的回复也添加到历史中
            self.dialogue_history.add_message("user", user_input)
            self.dialogue_history.add_message("assistant", replay_response)
            return replay_response
        
        # 添加用户消息到历史
        self.dialogue_history.add_message("user", user_input)
        
        # 处理任务型对话（可能涉及function calling）
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # 获取对话历史
            messages = self.dialogue_history.get_conversation_history()
            
            # 获取函数schema
            functions = self.function_caller.get_function_schemas() if self.config.get("enable_function_calling", True) else None
            
            # 调用模型
            logger.info(f"第 {iteration} 轮：发送请求到模型...")
            try:
                response = self.model_client.chat(messages, functions)
            except Exception as e:
                logger.error(f"模型调用失败：{e}")
                error_msg = f"抱歉，模型调用失败：{str(e)}"
                self.dialogue_history.add_message("assistant", error_msg)
                return error_msg
            
            # 检查是否有function call
            function_call = self.function_caller.parse_function_call(response)
            
            if function_call:
                # 有function call，执行函数
                logger.info(f"检测到function call：{function_call['function']['name']}")
                
                # 执行函数
                func_result = self.function_caller.execute_function_call(function_call)
                
                # 格式化结果并添加到历史
                formatted_result = self.function_caller.format_function_result(function_call, func_result)
                self.dialogue_history.add_message(**formatted_result)
                
                # 继续下一轮，让模型基于函数结果生成回复
                logger.info("函数执行完成，等待模型生成最终回复...")
                continue
            
            else:
                # 没有function call，返回模型回复
                assistant_message = response["choices"][0]["message"]
                content = assistant_message.get("content", "")
                
                # 添加到历史
                self.dialogue_history.add_message("assistant", content)
                
                return content
        
        error_msg = "抱歉，达到最大迭代次数，可能存在循环调用问题。"
        self.dialogue_history.add_message("assistant", error_msg)
        return error_msg
    
    def reset(self):
        """重置对话历史"""
        self.dialogue_history.clear_history()
        if self.config.get("system_prompt"):
            self.dialogue_history.add_message("system", self.config["system_prompt"])
        logger.info("对话历史已重置")
