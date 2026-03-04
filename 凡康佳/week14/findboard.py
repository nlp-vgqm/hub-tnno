"""
开发板查询agent，可以查询开发板信息
"""

import os
import json
from openai import OpenAI

# ===========================================
# 工具函数定义
# ==========

def get_baords_list():
    """获取开发板列表"""
    products = [
        {
            "id": "rv_001",
            "name": "香橙派RV2",
            "type": "riscv 开发板",
            "description": "香橙派开发的一款开发板"
        },
        {
            "id": "rv_002",
            "name": "香橙派RS2",
            "type": "riscv 开发板",
            "description": "适合做路由的高性能开发板"
        },
        {
            "id": "esp32_001",
            "name": "乐鑫ESP32",
            "type": "esp32开发板",
            "description": "适合做路由的esp32开发板"
        },
        {
            "id": "visionfive_001",
            "name": "昉星光开发版1代",
            "type": "risc v 开发板",
            "description": "扩展接口很多，深度适配各大linux发行版以及开源鸿蒙的开发板"
        },
        {
            "id": "visionfive_002",
            "name": "昉星光开发版2代",
            "type": "risc v 开发板",
            "description": "相比较第一代做了性能提升"
        }
    ]
    return json.dumps(products, ensure_ascii=False)

def get_board_detail(board_id: str):
    """获取开发板详情"""
    products = {
        'rv_001':{
            "id": "rv_001",
            "接口": ["USB-C", "USB-A", "RJ45","SD卡"],
            "内存规格":["DDR4 2G","DDR4 4G","DDR4 8G"],
            "存储规格": ["TF-eMMC-8G", "TF-eMMC-16G", "TF-eMMC-32G"],
            "核心数": 8
        },
        'rv_002':{
            "id": "rv_002",
            "接口": ["USB-C", "USB-A", "RJ45*4","SD卡"],
            "内存规格":["DDR4 2G","DDR4 4G","DDR4 8G"],
            "存储规格": ["TF-eMMC-8G", "TF-eMMC-16G", "TF-eMMC-32G"],
            "核心数": 8
        },
        'esp32_001':{
            "id": "esp32_001",
            "接口": ["USB-C","SD卡"],
            "内存规格":["DDR4 2G"],
            "存储规格": ["TF-eMMC-8G"],
            "核心数": 2
        },
        'visionfive_001':{
            "id": "visionfive_001",
            "接口": ["USB-C", "USB-A", "RJ45*2","SD卡",'M2','camera*2'],
            "内存规格":["DDR4 4G","DDR4 8G"],
            "存储规格": ["TF-eMMC-32G", "TF-eMMC-64G"],
            "核心数": 4
        },
        'visionfive_002':{
            "id": "visionfive_002",
            "接口": ["USB-C", "USB-A", "RJ45*2","SD卡",'M2','camera*2'],
            "内存规格":["DDR4 4G","DDR4 8G"],
            "存储规格": ["TF-eMMC-32G", "TF-eMMC-64G"],
            "核心数": 4
        }
    }

    return json.dumps(products[board_id], ensure_ascii=False)

def get_board_price(board_id: str):
    """获取开发板价格"""
    products = {
        'rv_001':{
            "id": "rv_001",
            "price": 300
        },
        'rv_002':{
            "id": "rv_002",
            "price": 200
        },
        'esp32_001':{
            "id": "esp32_001",
            "price": 50
        },
        'visionfive_001':{
            "id": "visionfive_001",
            "price": 500
        },
        'visionfive_002':{
            "id": "visionfive_002",
            "price": 700
        }
    }
    return json.dumps({"price": products[board_id]}, ensure_ascii=False)


def get_board_downlaod_url(board_id: str):
    """获取开发板支持材料下载链接"""
    return json.dumps({"url": "https://example.com/download/"+str(board_id)}, ensure_ascii=False)


# ===========================================
# 工具函数的JSON 定义

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_baords_list",
            "description": "获取所有开发板列表",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_board_detail",
            "description": "获取对应开发板的详细信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "board_id": {
                        "type": "string",
                        "description": "开发板ID"
                    }
                },
                "required": ["board_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_board_price",
            "description": "获取对应开发板的价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "board_id": {
                        "type": "string",
                        "description": "开发板ID"
                    }
                },
                "required": ["board_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_board_downlaod_url",
            "description": "获取对应开发板的资料下载链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "board_id": {
                        "type": "string",
                        "description": "开发板ID"
                    }
                },
                "required": ["board_id"]
            }
        }
    }
]


# ===========================================
# 工具函数映射
available_functions = {
    "get_baords_list": get_baords_list,
    "get_board_detail": get_board_detail,
    "get_board_price": get_board_price,
    "get_board_downlaod_url": get_board_downlaod_url
}

def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus") -> str:
    """运行开发板查询agent"""
    client = OpenAI(
        api_key=api_key ,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {"role": "system", "content": "你是一个帮助用户查询开发板的助手.你需要根据用户的查询，调用对应的工具函数，并给出结果。"},
        {"role": "user", "content": user_query}
    ]

    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )

        response_message = response.choices[0].message

        messages.append(response_message)

        tool_calls = response_message.tool_calls

        if not tool_calls:
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "content": function_response,
                    "tool_call_id": tool_call.id,
                    "name": function_name
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"



if __name__ == "__main__":
    # user_query = "请列出所有开发板，并给出每个开发板的价格"
    # user_query = "我想要一个适合做路由器的开发板，请列出最适合的一款，以及其详细信息还有资料下载链接"
    # user_query = "我想找一个最便宜的开发板"
    # user_query = "我有图像识别的需求，帮我找到最合适的开发板"
    user_query = "我想知道香橙派RV2和香橙派RS2有什么区别"
    api_key = ""
    model = "qwen-plus"

    result = run_agent(user_query, api_key, model)
    print(result)

