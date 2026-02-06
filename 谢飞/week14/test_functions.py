# -*- coding: utf-8 -*-

"""
测试函数
"""

from functions import (
    calculator,
    get_current_time,
    weather_query,
    text_processing,
    database_query,
    call_function
)


def test_calculator():
    """测试计算器"""
    print("=" * 60)
    print("测试计算器函数")
    print("=" * 60)
    
    test_cases = [
        "2 + 3",
        "10 * 5 - 3",
        "(10 + 5) / 3",
        "2 ** 3",
    ]
    
    for expr in test_cases:
        result = calculator(expr)
        print(f"表达式: {expr}")
        print(f"结果: {result}")
        print()


def test_time():
    """测试时间查询"""
    print("=" * 60)
    print("测试时间查询函数")
    print("=" * 60)
    
    result = get_current_time()
    print(f"结果: {result}")
    print()
    
    result = get_current_time(format_type="simple")
    print(f"简单格式: {result}")
    print()


def test_weather():
    """测试天气查询"""
    print("=" * 60)
    print("测试天气查询函数")
    print("=" * 60)
    
    cities = ["北京", "上海", "广州", "未知城市"]
    
    for city in cities:
        result = weather_query(city)
        print(f"城市: {city}")
        print(f"结果: {result}")
        print()


def test_text_processing():
    """测试文本处理"""
    print("=" * 60)
    print("测试文本处理函数")
    print("=" * 60)
    
    # 翻译
    result = text_processing("Hello", "translate", "zh")
    print(f"翻译结果: {result}")
    print()
    
    # 摘要
    long_text = "这是一个很长的文本，需要被摘要。它包含了很多信息，但是用户可能只需要关键信息。摘要功能可以帮助用户快速了解文本的主要内容。"
    result = text_processing(long_text, "summarize")
    print(f"摘要结果: {result}")
    print()
    
    # 关键词
    result = text_processing("人工智能和机器学习是计算机科学的重要分支", "keywords")
    print(f"关键词结果: {result}")
    print()


def test_database():
    """测试数据库查询"""
    print("=" * 60)
    print("测试数据库查询函数")
    print("=" * 60)
    
    # 列出表
    result = database_query("list_tables")
    print(f"列出表: {result}")
    print()
    
    # 查询数据
    result = database_query("select", table="users")
    print(f"查询users表: {result}")
    print()
    
    # 计数
    result = database_query("count", table="products")
    print(f"products表计数: {result}")
    print()


def test_function_caller():
    """测试函数调用器"""
    print("=" * 60)
    print("测试函数调用器")
    print("=" * 60)
    
    # 测试通过call_function调用
    result = call_function("calculator", {"expression": "5 * 6"})
    print(f"调用calculator: {result}")
    print()
    
    result = call_function("weather_query", {"city": "北京"})
    print(f"调用weather_query: {result}")
    print()
    
    # 测试不存在的函数
    result = call_function("nonexistent", {})
    print(f"调用不存在的函数: {result}")
    print()


if __name__ == "__main__":
    print("\n开始测试所有函数...\n")
    
    try:
        test_calculator()
        test_time()
        test_weather()
        test_text_processing()
        test_database()
        test_function_caller()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
    
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
