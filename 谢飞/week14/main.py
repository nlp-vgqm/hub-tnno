# -*- coding: utf-8 -*-

"""
基于大模型的Function Calling任务主程序

Function Calling允许大模型在对话过程中调用预定义的函数来完成特定任务。
本程序实现了完整的function calling流程：
1. 定义多个实用函数（计算器、天气查询、时间查询等）
2. 将函数schema发送给大模型
3. 模型根据用户输入决定是否调用函数
4. 执行函数并返回结果
5. 模型基于函数结果生成最终回复

使用方法：
1. 配置config.py（设置API密钥等）
2. 运行：python main.py
3. 在交互式对话中输入问题，模型会自动调用相应的函数
"""

import os
import json
import logging
from typing import Dict, Any, List

from config import Config
from model_client import create_model_client
from function_caller import FunctionCaller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FunctionCallingChat:
    """Function Calling聊天系统"""
    
    def __init__(self, config):
        self.config = config
        self.model_client = create_model_client(config)
        self.function_caller = FunctionCaller(config)
        
        # 初始化系统消息
        if config.get("system_prompt"):
            self.function_caller.add_to_history("system", config["system_prompt"])
    
    def chat(self, user_input: str, max_iterations: int = 5) -> str:
        """
        处理用户输入，支持多轮function calling
        
        Args:
            user_input: 用户输入
            max_iterations: 最大迭代次数（防止无限循环）
        
        Returns:
            最终回复
        """
        # 添加用户消息
        self.function_caller.add_to_history("user", user_input)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # 获取对话历史
            messages = self.function_caller.get_conversation_history()
            
            # 获取函数schema
            functions = self.function_caller.get_function_schemas() if self.config.get("enable_function_calling", True) else None
            
            # 调用模型
            logger.info(f"第 {iteration} 轮：发送请求到模型...")
            try:
                response = self.model_client.chat(messages, functions)
            except Exception as e:
                logger.error(f"模型调用失败：{e}")
                return f"抱歉，模型调用失败：{str(e)}"
            
            # 检查是否有function call
            function_call = self.function_caller.parse_function_call(response)
            
            if function_call:
                # 有function call，执行函数
                logger.info(f"检测到function call：{function_call['function']['name']}")
                
                # 执行函数
                func_result = self.function_caller.execute_function_call(function_call)
                
                # 格式化结果并添加到历史
                formatted_result = self.function_caller.format_function_result(function_call, func_result)
                self.function_caller.add_to_history(**formatted_result)
                
                # 继续下一轮，让模型基于函数结果生成回复
                logger.info("函数执行完成，等待模型生成最终回复...")
                continue
            
            else:
                # 没有function call，返回模型回复
                assistant_message = response["choices"][0]["message"]
                content = assistant_message.get("content", "")
                
                # 添加到历史
                self.function_caller.add_to_history("assistant", content)
                
                return content
        
        return "抱歉，达到最大迭代次数，可能存在循环调用问题。"
    
    def reset(self):
        """重置对话历史"""
        self.function_caller.clear_history()
        if self.config.get("system_prompt"):
            self.function_caller.add_to_history("system", self.config["system_prompt"])
        logger.info("对话历史已重置")


def interactive_chat(config: Dict[str, Any]):
    """交互式聊天"""
    chat = FunctionCallingChat(config)
    
    print("=" * 60)
    print("Function Calling 聊天系统")
    print("=" * 60)
    print("支持的功能：")
    print("  - 计算器：'计算 2 + 3 * 4'")
    print("  - 时间查询：'现在几点了？'")
    print("  - 天气查询：'北京天气怎么样？'")
    print("  - 文本处理：'翻译：Hello' 或 '摘要：...'")
    print("  - 数据库查询：'查询users表'")
    print("输入 'quit' 或 'exit' 退出，输入 'reset' 重置对话")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            if user_input.lower() in ["reset", "重置"]:
                chat.reset()
                print("对话已重置")
                continue
            
            # 处理用户输入
            response = chat.chat(user_input)
            print(f"助手: {response}")
            print()
        
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"处理错误：{e}")
            print(f"抱歉，发生了错误：{str(e)}")
            print()


def demo_chat(config: Dict[str, Any]):
    """演示模式：运行预设的测试用例"""
    chat = FunctionCallingChat(config)
    
    test_cases = [
        "计算 15 * 8 + 20",
        "现在几点了？",
        "北京今天天气怎么样？",
        "查询数据库中有哪些表？",
        "翻译：Hello",
    ]
    
    print("=" * 60)
    print("Function Calling 演示模式")
    print("=" * 60)
    print()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_input}")
        print("-" * 60)
        
        try:
            response = chat.chat(test_input)
            print(f"助手: {response}")
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print()
        print("=" * 60)
        print()
        
        # 重置对话历史（可选）
        # chat.reset()


def main():
    """主函数"""
    config = Config
    
    # 检查配置
    if config["model_type"] == "openai":
        if not config["openai_api_key"]:
            logger.error("请设置OPENAI_API_KEY环境变量或在config.py中配置")
            logger.error("例如：export OPENAI_API_KEY='your-api-key'")
            return
    
    print("=" * 60)
    print("基于大模型的Function Calling任务")
    print("=" * 60)
    print(f"模型类型: {config['model_type']}")
    if config["model_type"] == "openai":
        print(f"模型: {config['openai_model']}")
    elif config["model_type"] == "local":
        print(f"模型路径: {config['local_model_path']}")
    print("=" * 60)
    print()
    
    # 选择模式
    mode = input("选择模式 (1=交互式, 2=演示): ").strip()
    
    if mode == "1":
        interactive_chat(config)
    elif mode == "2":
        demo_chat(config)
    else:
        print("无效选择，使用演示模式")
        demo_chat(config)


if __name__ == "__main__":
    main()
