# -*- coding: utf-8 -*-

"""
对话历史管理器
负责管理对话历史记录，支持重听功能
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DialogueHistory:
    """对话历史管理器"""
    
    def __init__(self, config):
        self.config = config
        self.conversation_history = []  # 完整的对话历史
        self.assistant_responses = []  # 助手回复历史（用于重听功能）
    
    def add_message(self, role: str, content: Any, name: Optional[str] = None):
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色（system, user, assistant, tool）
            content: 消息内容
            name: 可选的名字（用于tool消息）
        """
        message = {"role": role}
        
        if name:
            message["name"] = name
        
        if isinstance(content, dict):
            message.update(content)
        else:
            message["content"] = content
        
        self.conversation_history.append(message)
        
        # 如果是助手回复，保存到重听历史
        if role == "assistant" and "content" in message:
            self.assistant_responses.append({
                "content": message["content"],
                "index": len(self.assistant_responses)
            })
            logger.debug(f"保存助手回复到重听历史，当前共 {len(self.assistant_responses)} 条")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        获取完整的对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history
    
    def get_last_assistant_response(self) -> Optional[str]:
        """
        获取最后一次助手回复（用于重听功能）
        
        Returns:
            最后一次助手回复内容，如果没有则返回None
        """
        if self.assistant_responses:
            return self.assistant_responses[-1]["content"]
        return None
    
    def get_assistant_response_by_index(self, index: int) -> Optional[str]:
        """
        根据索引获取助手回复
        
        Args:
            index: 回复索引（从0开始，-1表示最后一次）
        
        Returns:
            助手回复内容，如果索引无效则返回None
        """
        if index < 0:
            index = len(self.assistant_responses) + index
        
        if 0 <= index < len(self.assistant_responses):
            return self.assistant_responses[index]["content"]
        return None
    
    def get_recent_responses(self, count: int = 3) -> List[str]:
        """
        获取最近的N条助手回复
        
        Args:
            count: 要获取的回复数量
        
        Returns:
            最近的助手回复列表
        """
        if not self.assistant_responses:
            return []
        
        start_idx = max(0, len(self.assistant_responses) - count)
        return [resp["content"] for resp in self.assistant_responses[start_idx:]]
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.assistant_responses = []
        logger.info("对话历史已清空")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        获取对话历史摘要
        
        Returns:
            包含历史统计信息的字典
        """
        return {
            "total_messages": len(self.conversation_history),
            "total_assistant_responses": len(self.assistant_responses),
            "has_recent_responses": len(self.assistant_responses) > 0
        }
