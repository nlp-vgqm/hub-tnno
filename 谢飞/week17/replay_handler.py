# -*- coding: utf-8 -*-

"""
重听功能处理器
处理用户的重听请求，返回之前的助手回复
"""

import re
import logging
from typing import Optional, Dict, Any, List
from dialogue_history import DialogueHistory

logger = logging.getLogger(__name__)


class ReplayHandler:
    """重听功能处理器"""
    
    def __init__(self, config, dialogue_history: DialogueHistory):
        self.config = config
        self.dialogue_history = dialogue_history
        self.replay_keywords = config.get("replay_keywords", ["重听", "再说一遍", "重复", "再说一次"])
        self.enable_replay = config.get("enable_replay", True)
    
    def is_replay_request(self, user_input: str) -> bool:
        """
        判断用户输入是否是重听请求
        
        Args:
            user_input: 用户输入文本
        
        Returns:
            如果是重听请求返回True，否则返回False
        """
        if not self.enable_replay:
            return False
        
        # 转换为小写进行匹配
        user_input_lower = user_input.lower().strip()
        
        # 检查是否包含重听关键词
        for keyword in self.replay_keywords:
            if keyword.lower() in user_input_lower:
                return True
        
        # 检查是否匹配重听模式（如"重复上一句"、"再说一遍刚才的话"等）
        replay_patterns = [
            r"重复.*(?:上|刚才|之前|刚才的|上一次)",
            r"再说.*(?:一遍|一次|一次刚才|一遍刚才)",
            r"重.*(?:说|复|播)",
            r"刚才.*(?:说|讲|回答)",
        ]
        
        for pattern in replay_patterns:
            if re.search(pattern, user_input_lower):
                return True
        
        return False
    
    def extract_replay_index(self, user_input: str) -> Optional[int]:
        """
        从用户输入中提取要重听的回复索引
        
        Args:
            user_input: 用户输入文本
        
        Returns:
            回复索引（从0开始，-1表示最后一次），如果无法提取则返回None
        """
        # 默认返回最后一次回复
        default_index = -1
        
        # 尝试提取数字（如"重复第2条"、"重听第3个回复"）
        number_patterns = [
            r"第\s*(\d+)\s*(?:条|个|次|句|条回复|个回复)",
            r"(\d+)\s*(?:条|个|次|句)",
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, user_input)
            if match:
                try:
                    index = int(match.group(1)) - 1  # 转换为0-based索引
                    if index >= 0:
                        return index
                except ValueError:
                    pass
        
        # 检查是否指定了"上一次"、"上一条"等
        if re.search(r"(?:上|前)\s*(?:一|1)\s*(?:条|个|次|句)", user_input):
            return -2  # 倒数第二条
        
        return default_index
    
    def handle_replay_request(self, user_input: str) -> Optional[str]:
        """
        处理重听请求
        
        Args:
            user_input: 用户输入文本
        
        Returns:
            要重听的回复内容，如果没有可重听的内容则返回None
        """
        if not self.is_replay_request(user_input):
            return None
        
        # 提取索引
        index = self.extract_replay_index(user_input)
        
        # 获取对应的回复
        if index is None:
            response = self.dialogue_history.get_last_assistant_response()
        else:
            response = self.dialogue_history.get_assistant_response_by_index(index)
        
        if response:
            logger.info(f"重听请求：返回索引 {index} 的回复")
            return response
        else:
            logger.warning("重听请求：没有可重听的内容")
            return None
    
    def get_replay_suggestions(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        获取可重听的回复建议
        
        Args:
            count: 要获取的建议数量
        
        Returns:
            回复建议列表，每个元素包含索引和内容预览
        """
        recent_responses = self.dialogue_history.get_recent_responses(count)
        suggestions = []
        
        for i, response in enumerate(recent_responses):
            # 截取前50个字符作为预览
            preview = response[:50] + "..." if len(response) > 50 else response
            suggestions.append({
                "index": len(recent_responses) - count + i,
                "preview": preview,
                "full_content": response
            })
        
        return suggestions
