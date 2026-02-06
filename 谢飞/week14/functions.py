# -*- coding: utf-8 -*-

"""
自定义函数定义
用于Function Calling任务的各种工具函数
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List


def calculator(expression: str) -> Dict[str, Any]:
    """
    计算器函数：执行数学表达式计算
    
    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4", "(10 + 5) / 3"
    
    Returns:
        包含计算结果和表达式的字典
    """
    try:
        # 安全检查：只允许数字、运算符和括号
        if not re.match(r'^[0-9+\-*/().\s]+$', expression):
            return {
                "success": False,
                "result": None,
                "error": "表达式包含非法字符",
                "expression": expression
            }
        
        # 执行计算
        result = eval(expression)
        
        return {
            "success": True,
            "result": result,
            "expression": expression,
            "message": f"计算结果：{expression} = {result}"
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "expression": expression,
            "message": f"计算失败：{str(e)}"
        }


def get_current_time(timezone: str = "Asia/Shanghai", format_type: str = "full") -> Dict[str, Any]:
    """
    获取当前时间
    
    Args:
        timezone: 时区，默认"Asia/Shanghai"
        format_type: 格式类型，"full"（完整）或"simple"（简单）
    
    Returns:
        包含当前时间信息的字典
    """
    try:
        now = datetime.now()
        
        if format_type == "full":
            time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
            weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
            time_info = f"{time_str} {weekday}"
        else:
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            time_info = time_str
        
        return {
            "success": True,
            "time": time_info,
            "timestamp": now.timestamp(),
            "timezone": timezone,
            "format": format_type,
            "message": f"当前时间：{time_info}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"获取时间失败：{str(e)}"
        }


def weather_query(city: str, date: str = None) -> Dict[str, Any]:
    """
    查询天气信息（模拟函数）
    
    Args:
        city: 城市名称，例如 "北京", "上海"
        date: 日期，格式"YYYY-MM-DD"，默认为今天
    
    Returns:
        包含天气信息的字典
    """
    # 模拟天气数据
    mock_weather_data = {
        "北京": {"temp": "15°C", "condition": "晴", "humidity": "45%", "wind": "微风"},
        "上海": {"temp": "18°C", "condition": "多云", "humidity": "60%", "wind": "东南风2级"},
        "广州": {"temp": "25°C", "condition": "晴", "humidity": "70%", "wind": "南风1级"},
        "深圳": {"temp": "26°C", "condition": "多云", "humidity": "68%", "wind": "南风2级"},
        "杭州": {"temp": "16°C", "condition": "小雨", "humidity": "75%", "wind": "东北风1级"},
    }
    
    try:
        # 查找城市（支持模糊匹配）
        city_key = None
        for key in mock_weather_data.keys():
            if city in key or key in city:
                city_key = key
                break
        
        if city_key is None:
            # 返回默认天气
            weather_info = {"temp": "20°C", "condition": "晴", "humidity": "50%", "wind": "微风"}
            city_key = city
        else:
            weather_info = mock_weather_data[city_key]
        
        date_str = date if date else datetime.now().strftime("%Y-%m-%d")
        
        return {
            "success": True,
            "city": city_key,
            "date": date_str,
            "temperature": weather_info["temp"],
            "condition": weather_info["condition"],
            "humidity": weather_info["humidity"],
            "wind": weather_info["wind"],
            "message": f"{city_key} {date_str} 天气：{weather_info['condition']}，温度{weather_info['temp']}，湿度{weather_info['humidity']}，{weather_info['wind']}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"查询天气失败：{str(e)}"
        }


def text_processing(text: str, operation: str, target_language: str = "en") -> Dict[str, Any]:
    """
    文本处理函数：支持翻译、摘要、关键词提取等
    
    Args:
        text: 要处理的文本
        operation: 操作类型，"translate"（翻译）、"summarize"（摘要）、"keywords"（关键词）
        target_language: 目标语言（仅用于翻译），"en"（英语）、"zh"（中文）、"ja"（日语）
    
    Returns:
        包含处理结果的字典
    """
    try:
        if operation == "translate":
            # 简单的翻译模拟（实际应该调用翻译API）
            translations = {
                ("zh", "en"): {
                    "你好": "Hello",
                    "谢谢": "Thank you",
                    "再见": "Goodbye"
                },
                ("en", "zh"): {
                    "Hello": "你好",
                    "Thank you": "谢谢",
                    "Goodbye": "再见"
                }
            }
            
            # 简单的关键词匹配翻译
            result = text  # 默认返回原文
            for key, value in translations.get((target_language, "en") if target_language == "en" else ("en", target_language), {}).items():
                if key in text:
                    result = value
                    break
            
            return {
                "success": True,
                "operation": "translate",
                "original_text": text,
                "translated_text": result,
                "target_language": target_language,
                "message": f"翻译结果：{result}"
            }
        
        elif operation == "summarize":
            # 简单的摘要模拟
            words = text.split()
            summary_length = min(20, len(words))
            summary = " ".join(words[:summary_length])
            if len(words) > summary_length:
                summary += "..."
            
            return {
                "success": True,
                "operation": "summarize",
                "original_text": text,
                "summary": summary,
                "message": f"摘要：{summary}"
            }
        
        elif operation == "keywords":
            # 简单的关键词提取（提取长度>=2的词）
            words = re.findall(r'\b\w{2,}\b', text)
            keywords = list(set(words))[:5]  # 最多5个关键词
            
            return {
                "success": True,
                "operation": "keywords",
                "original_text": text,
                "keywords": keywords,
                "message": f"关键词：{', '.join(keywords)}"
            }
        
        else:
            return {
                "success": False,
                "error": f"不支持的操作类型：{operation}",
                "message": f"支持的操作：translate, summarize, keywords"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"文本处理失败：{str(e)}"
        }


def database_query(query_type: str, table: str = None, condition: str = None) -> Dict[str, Any]:
    """
    数据库查询函数（模拟）
    
    Args:
        query_type: 查询类型，"select"（查询）、"count"（计数）、"list_tables"（列出表）
        table: 表名
        condition: 查询条件
    
    Returns:
        包含查询结果的字典
    """
    # 模拟数据库
    mock_database = {
        "users": [
            {"id": 1, "name": "张三", "age": 25, "city": "北京"},
            {"id": 2, "name": "李四", "age": 30, "city": "上海"},
            {"id": 3, "name": "王五", "age": 28, "city": "广州"},
        ],
        "products": [
            {"id": 1, "name": "笔记本电脑", "price": 5999, "stock": 50},
            {"id": 2, "name": "手机", "price": 3999, "stock": 100},
            {"id": 3, "name": "平板电脑", "price": 2999, "stock": 30},
        ]
    }
    
    try:
        if query_type == "list_tables":
            tables = list(mock_database.keys())
            return {
                "success": True,
                "tables": tables,
                "message": f"数据库表：{', '.join(tables)}"
            }
        
        elif query_type == "select":
            if table not in mock_database:
                return {
                    "success": False,
                    "error": f"表 '{table}' 不存在",
                    "message": f"可用的表：{', '.join(mock_database.keys())}"
                }
            
            data = mock_database[table]
            
            # 简单的条件过滤（实际应该用SQL）
            if condition:
                # 示例：condition = "age > 25"
                filtered_data = data
                # 这里只是简单示例，实际应该解析SQL条件
            else:
                filtered_data = data
            
            return {
                "success": True,
                "table": table,
                "count": len(filtered_data),
                "data": filtered_data,
                "message": f"查询到 {len(filtered_data)} 条记录"
            }
        
        elif query_type == "count":
            if table not in mock_database:
                return {
                    "success": False,
                    "error": f"表 '{table}' 不存在"
                }
            
            count = len(mock_database[table])
            return {
                "success": True,
                "table": table,
                "count": count,
                "message": f"表 '{table}' 共有 {count} 条记录"
            }
        
        else:
            return {
                "success": False,
                "error": f"不支持的查询类型：{query_type}",
                "message": "支持的查询类型：select, count, list_tables"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"数据库查询失败：{str(e)}"
        }


# 函数注册表：定义所有可用的函数及其schema
FUNCTIONS = [
    {
        "name": "calculator",
        "description": "执行数学表达式计算。支持加减乘除和括号运算。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2 + 3 * 4' 或 '(10 + 5) / 3'"
                }
            },
            "required": ["expression"]
        },
        "function": calculator
    },
    {
        "name": "get_current_time",
        "description": "获取当前时间和日期信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，默认为'Asia/Shanghai'",
                    "default": "Asia/Shanghai"
                },
                "format_type": {
                    "type": "string",
                    "description": "时间格式类型，'full'（完整格式）或'simple'（简单格式）",
                    "enum": ["full", "simple"],
                    "default": "full"
                }
            },
            "required": []
        },
        "function": get_current_time
    },
    {
        "name": "weather_query",
        "description": "查询指定城市的天气信息，包括温度、天气状况、湿度和风力。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如 '北京'、'上海'、'广州'"
                },
                "date": {
                    "type": "string",
                    "description": "日期，格式为'YYYY-MM-DD'，默认为今天"
                }
            },
            "required": ["city"]
        },
        "function": weather_query
    },
    {
        "name": "text_processing",
        "description": "处理文本，支持翻译、摘要和关键词提取。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要处理的文本内容"
                },
                "operation": {
                    "type": "string",
                    "description": "操作类型：'translate'（翻译）、'summarize'（摘要）、'keywords'（关键词提取）",
                    "enum": ["translate", "summarize", "keywords"]
                },
                "target_language": {
                    "type": "string",
                    "description": "目标语言（仅用于翻译），'en'（英语）、'zh'（中文）、'ja'（日语）",
                    "default": "en"
                }
            },
            "required": ["text", "operation"]
        },
        "function": text_processing
    },
    {
        "name": "database_query",
        "description": "查询数据库信息，支持查询数据、计数和列出表名。",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "description": "查询类型：'select'（查询数据）、'count'（计数）、'list_tables'（列出所有表）",
                    "enum": ["select", "count", "list_tables"]
                },
                "table": {
                    "type": "string",
                    "description": "表名，例如 'users'、'products'"
                },
                "condition": {
                    "type": "string",
                    "description": "查询条件（可选）"
                }
            },
            "required": ["query_type"]
        },
        "function": database_query
    }
]


def get_function_schemas() -> List[Dict[str, Any]]:
    """
    获取所有函数的schema（用于发送给大模型）
    
    Returns:
        函数schema列表
    """
    schemas = []
    for func_info in FUNCTIONS:
        schemas.append({
            "name": func_info["name"],
            "description": func_info["description"],
            "parameters": func_info["parameters"]
        })
    return schemas


def call_function(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用指定的函数
    
    Args:
        function_name: 函数名称
        arguments: 函数参数
    
    Returns:
        函数执行结果
    """
    for func_info in FUNCTIONS:
        if func_info["name"] == function_name:
            func = func_info["function"]
            try:
                result = func(**arguments)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"函数执行失败：{str(e)}"
                }
    
    return {
        "success": False,
        "error": f"函数 '{function_name}' 不存在",
        "message": f"可用的函数：{', '.join([f['name'] for f in FUNCTIONS])}"
    }
