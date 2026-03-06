# -*- coding: utf-8 -*-

"""
任务型对话系统主程序
支持Function Calling和重听功能
业务逻辑已拆分为多个模块文件
"""

import os
import logging
from typing import Dict, Any

from config import Config
from model_client import create_model_client
from dialogue_history import DialogueHistory
from function_caller import FunctionCaller
from replay_handler import ReplayHandler
from task_handler import TaskHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def interactive_chat(config: Dict[str, Any]):
    """交互式聊天"""
    # 初始化各个模块
    model_client = create_model_client(config)
    dialogue_history = DialogueHistory(config)
    function_caller = FunctionCaller(config)
    replay_handler = ReplayHandler(config, dialogue_history)
    task_handler = TaskHandler(config, model_client, dialogue_history, function_caller, replay_handler)
    
    # 初始化系统消息
    if config.get("system_prompt"):
        dialogue_history.add_message("system", config["system_prompt"])
    
    print("=" * 60)
    print("任务型对话系统（支持重听功能）")
    print("=" * 60)
    print("支持的功能：")
    print("  - 计算器：'计算 2 + 3 * 4'")
    print("  - 时间查询：'现在几点了？'")
    print("  - 天气查询：'北京天气怎么样？'")
    print("  - 文本处理：'翻译：Hello' 或 '摘要：...'")
    print("  - 数据库查询：'查询users表'")
    print("  - 重听功能：'重听'、'再说一遍'、'重复'")
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
                task_handler.reset()
                print("对话已重置")
                continue
            
            # 处理用户输入
            response = task_handler.process_user_input(user_input)
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
    # 初始化各个模块
    model_client = create_model_client(config)
    dialogue_history = DialogueHistory(config)
    function_caller = FunctionCaller(config)
    replay_handler = ReplayHandler(config, dialogue_history)
    task_handler = TaskHandler(config, model_client, dialogue_history, function_caller, replay_handler)
    
    # 初始化系统消息
    if config.get("system_prompt"):
        dialogue_history.add_message("system", config["system_prompt"])
    
    test_cases = [
        "计算 15 * 8 + 20",
        "现在几点了？",
        "北京今天天气怎么样？",
        "查询数据库中有哪些表？",
        "翻译：Hello",
        "重听",  # 测试重听功能
    ]
    
    print("=" * 60)
    print("任务型对话系统演示模式")
    print("=" * 60)
    print()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_input}")
        print("-" * 60)
        
        try:
            response = task_handler.process_user_input(test_input)
            print(f"助手: {response}")
        except Exception as e:
            print(f"错误: {str(e)}")
        
        print()
        print("=" * 60)
        print()


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
    print("任务型对话系统（第十七周作业）")
    print("=" * 60)
    print("功能特性：")
    print("  - 任务型对话（Function Calling）")
    print("  - 重听功能（支持重复之前的回复）")
    print("  - 业务逻辑多文件拆分")
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
