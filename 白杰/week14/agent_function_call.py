"""
银行贷款Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程（核心：对比等额本金/等额本息）
"""

import os
import json
from openai import OpenAI

# ==================== 工具函数定义（银行贷款领域） ====================
# 核心：实现等额本金、等额本息的计算与对比

def get_loan_repayment_methods():
    """
    获取所有可用的贷款还款方式列表
    """
    methods = [
        {
            "id": "equal_principal",
            "name": "等额本金",
            "description": "每月偿还固定本金+剩余本金产生的利息，还款额逐月递减",
            "feature": "前期还款压力大，后期压力小，总利息更少",
            "suitable_for": "收入稳定且偏高，想节省总利息的借款人"
        },
        {
            "id": "equal_installment",
            "name": "等额本息",
            "description": "每月偿还固定金额（本金+利息），还款额全程不变",
            "feature": "前期还款压力小，资金规划方便，总利息更多",
            "suitable_for": "收入稳定但偏低，需要平稳资金规划的借款人"
        }
    ]
    return json.dumps(methods, ensure_ascii=False)

def get_repayment_detail(method_id: str):
    """
    获取指定还款方式的详细信息

    Args:
        method_id: 还款方式ID，例如：equal_principal, equal_installment
    """
    details = {
        "equal_principal": {
            "id": "equal_principal",
            "name": "等额本金",
            "description": "又称“递减还款法”，每月偿还固定数额的本金和剩余本金在当月产生的利息",
            "core_feature": [
                "每月应还本金固定：贷款总额 ÷ 还款总月数",
                "每月应还利息递减：剩余本金 × 月利率",
                "每月总还款额逐月递减",
                "前期还款压力大，后期逐渐减轻",
                "总利息支出低于等额本息"
            ],
            "suitable_crowd": [
                "收入较高且稳定，能承受前期较高还款压力",
                "计划提前还款（前期还本金更多，提前还款更划算）",
                "注重节省总利息成本的借款人"
            ],
            "note": "适合30-45岁职场黄金期，收入呈上升趋势的人群"
        },
        "equal_installment": {
            "id": "equal_installment",
            "name": "等额本息",
            "description": "又称“等额还款法”，每月偿还固定数额的还款额（包含本金和利息）",
            "core_feature": [
                "每月还款额固定，方便资金规划和预算管理",
                "前期偿还利息占比高，本金占比低",
                "后期偿还本金占比高，利息占比低",
                "还款压力平稳，无明显前期压力",
                "总利息支出高于等额本金"
            ],
            "suitable_crowd": [
                "收入稳定但偏低，无法承受前期较高还款压力",
                "刚步入职场，收入呈稳步上升趋势的年轻人",
                "需要精准规划每月开支，避免后期还款压力的借款人"
            ],
            "note": "适合25-35岁，资金储备较少，注重生活稳定性的人群"
        }
    }

    if method_id in details:
        return json.dumps(details[method_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "还款方式不存在"}, ensure_ascii=False)

def calculate_equal_principal(loan_amount: int, years: int, age: int, annual_rate: float = 3.85):
    """
    计算等额本金还款方式的详细数据（银行标准公式）

    Args:
        loan_amount: 贷款金额（元）
        years: 贷款年限（年）
        age: 借款人年龄（岁，用于微调利率，30-40岁为优质客户，基准利率）
        annual_rate: 贷款年利率（%，默认3.85%，5年以上商业贷款基准利率）
    """
    # 边界值校验
    if loan_amount <= 0 or years <= 0 or age <= 0:
        return json.dumps({"error": "贷款金额、年限、年龄必须为正数"}, ensure_ascii=False)

    # 基础参数转换
    months = years * 12  # 还款总月数
    monthly_rate = annual_rate / 100 / 12  # 月利率
    fixed_principal_per_month = round(loan_amount / months, 2)  # 每月固定偿还本金

    # 年龄对利率的微调（30-40岁优质客户，基准利率；超出区间上浮0.1%，简化逻辑）
    if not (30 <= age <= 40):
        annual_rate += 0.1
        monthly_rate = annual_rate / 100 / 12
        note = f"因借款人年龄{age}岁，超出30-40岁优质客户区间，年利率上浮0.1%，当前年利率为{annual_rate}%"
    else:
        note = f"借款人年龄{age}岁，属于优质客户，执行基准利率{annual_rate}%"

    # 计算首月还款额、末月还款额、总利息
    first_month_interest = round(loan_amount * monthly_rate, 2)  # 首月利息
    first_month_payment = round(fixed_principal_per_month + first_month_interest, 2)  # 首月总还款额

    last_month_interest = round(fixed_principal_per_month * monthly_rate, 2)  # 末月利息
    last_month_payment = round(fixed_principal_per_month + last_month_interest, 2)  # 末月总还款额

    total_interest = round((months + 1) * loan_amount * monthly_rate / 2, 2)  # 等额本金总利息公式
    total_payment = round(loan_amount + total_interest, 2)  # 总还款额

    # 整理结果
    result = {
        "repayment_method": "等额本金",
        "method_id": "equal_principal",
        "loan_params": {
            "loan_amount": loan_amount,
            "loan_years": years,
            "borrower_age": age,
            "annual_interest_rate": f"{annual_rate}%",
            "monthly_interest_rate": f"{round(monthly_rate*100, 4)}%"
        },
        "key_data": {
            "monthly_fixed_principal": fixed_principal_per_month,
            "first_month_payment": first_month_payment,
            "last_month_payment": last_month_payment,
            "total_interest": total_interest,
            "total_payment": total_payment,
            "repayment_months": months
        },
        "feature": "每月还款额逐月递减，前期压力大，后期压力小，总利息更少",
        "note": note
    }

    return json.dumps(result, ensure_ascii=False)

def calculate_equal_installment(loan_amount: int, years: int, age: int, annual_rate: float = 3.85):
    """
    计算等额本息还款方式的详细数据（银行标准公式）

    Args:
        loan_amount: 贷款金额（元）
        years: 贷款年限（年）
        age: 借款人年龄（岁，用于微调利率）
        annual_rate: 贷款年利率（%，默认3.85%，5年以上商业贷款基准利率）
    """
    # 边界值校验
    if loan_amount <= 0 or years <= 0 or age <= 0:
        return json.dumps({"error": "贷款金额、年限、年龄必须为正数"}, ensure_ascii=False)

    # 基础参数转换
    months = years * 12  # 还款总月数
    monthly_rate = annual_rate / 100 / 12  # 月利率

    # 年龄对利率的微调（30-40岁优质客户，基准利率；超出区间上浮0.1%）
    if not (30 <= age <= 40):
        annual_rate += 0.1
        monthly_rate = annual_rate / 100 / 12
        note = f"因借款人年龄{age}岁，超出30-40岁优质客户区间，年利率上浮0.1%，当前年利率为{annual_rate}%"
    else:
        note = f"借款人年龄{age}岁，属于优质客户，执行基准利率{annual_rate}%"

    # 等额本息核心公式计算
    if monthly_rate == 0:
        monthly_payment = round(loan_amount / months, 2)
        total_interest = 0
    else:
        monthly_payment = round(
            loan_amount * monthly_rate * (1 + monthly_rate) ** months /
            ((1 + monthly_rate) ** months - 1),
            2
        )  # 每月固定还款额
        total_interest = round(monthly_payment * months - loan_amount, 2)  # 总利息

    total_payment = round(loan_amount + total_interest, 2)  # 总还款额

    # 整理结果
    result = {
        "repayment_method": "等额本息",
        "method_id": "equal_installment",
        "loan_params": {
            "loan_amount": loan_amount,
            "loan_years": years,
            "borrower_age": age,
            "annual_interest_rate": f"{annual_rate}%",
            "monthly_interest_rate": f"{round(monthly_rate*100, 4)}%"
        },
        "key_data": {
            "monthly_fixed_payment": monthly_payment,
            "total_interest": total_interest,
            "total_payment": total_payment,
            "repayment_months": months
        },
        "feature": "每月还款额固定，资金规划方便，前期压力小，总利息更多",
        "note": note
    }

    return json.dumps(result, ensure_ascii=False)

def compare_repayment_methods(loan_amount: int, years: int, age: int, annual_rate: float = 3.85):
    """
    对比等额本金和等额本息两种还款方式（核心需求：满足100万、20年、30岁的对比）

    Args:
        loan_amount: 贷款金额（元）
        years: 贷款年限（年）
        age: 借款人年龄（岁）
        annual_rate: 贷款年利率（%，默认3.85%）
    """
    # 分别计算两种还款方式的结果
    equal_principal_result = json.loads(calculate_equal_principal(loan_amount, years, age, annual_rate))
    equal_installment_result = json.loads(calculate_equal_installment(loan_amount, years, age, annual_rate))

    # 校验是否有错误
    if "error" in equal_principal_result or "error" in equal_installment_result:
        error_msg = equal_principal_result.get("error", "") or equal_installment_result.get("error", "")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    # 整理对比核心数据（突出差异）
    comparison = {
        "compare_params": {
            "loan_amount": loan_amount,
            "loan_years": years,
            "borrower_age": age,
            "annual_interest_rate": f"{annual_rate}%"
        },
        "two_methods": [
            {
                "method_name": "等额本金",
                "monthly_payment": f"首月{equal_principal_result['key_data']['first_month_payment']}元，末月{equal_principal_result['key_data']['last_month_payment']}元（逐月递减）",
                "total_interest": equal_principal_result['key_data']['total_interest'],
                "total_payment": equal_principal_result['key_data']['total_payment'],
                "advantage": "总利息更少，后期还款压力小，适合提前还款",
                "disadvantage": "前期还款压力大，资金规划灵活性低"
            },
            {
                "method_name": "等额本息",
                "monthly_payment": f"每月固定{equal_installment_result['key_data']['monthly_fixed_payment']}元",
                "total_interest": equal_installment_result['key_data']['total_interest'],
                "total_payment": equal_installment_result['key_data']['total_payment'],
                "advantage": "每月还款固定，资金规划方便，前期压力小",
                "disadvantage": "总利息更多，前期还本金少，提前还款不划算"
            }
        ],
        "core_difference": f"等额本金比等额本息节省利息{round(equal_installment_result['key_data']['total_interest'] - equal_principal_result['key_data']['total_interest'], 2)}元，但其首月还款额比等额本息高{round(equal_principal_result['key_data']['first_month_payment'] - equal_installment_result['key_data']['monthly_fixed_payment'], 2)}元",
        "suggestion": "若你收入较高且稳定，能承受前期压力，优先选择等额本金（节省总利息）；若你需要平稳的资金规划，避免前期高压力，优先选择等额本息"
    }

    return json.dumps(comparison, ensure_ascii=False)

# ==================== 工具函数的JSON Schema定义 ====================
# 对应工具函数，供LLM识别和调用
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_loan_repayment_methods",
            "description": "获取所有可用的银行贷款还款方式列表，包括名称、特点、适用人群等基本信息",
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
            "name": "get_repayment_detail",
            "description": "获取指定贷款还款方式的详细信息，包括核心特征、适用人群、注意事项等",
            "parameters": {
                "type": "object",
                "properties": {
                    "method_id": {
                        "type": "string",
                        "description": "还款方式ID，例如：equal_principal（等额本金）、equal_installment（等额本息）"
                    }
                },
                "required": ["method_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_equal_principal",
            "description": "计算等额本金还款方式的详细数据，包括每月固定本金、首/末月还款额、总利息、总还款额等",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "integer",
                        "description": "贷款金额（元），例如：1000000"
                    },
                    "years": {
                        "type": "integer",
                        "description": "贷款年限（年），例如：20"
                    },
                    "age": {
                        "type": "integer",
                        "description": "借款人年龄（岁），例如：30"
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "贷款年利率（%），默认3.85%，可选参数"
                    }
                },
                "required": ["loan_amount", "years", "age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_equal_installment",
            "description": "计算等额本息还款方式的详细数据，包括每月固定还款额、总利息、总还款额等",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "integer",
                        "description": "贷款金额（元），例如：1000000"
                    },
                    "years": {
                        "type": "integer",
                        "description": "贷款年限（年），例如：20"
                    },
                    "age": {
                        "type": "integer",
                        "description": "借款人年龄（岁），例如：30"
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "贷款年利率（%），默认3.85%，可选参数"
                    }
                },
                "required": ["loan_amount", "years", "age"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_repayment_methods",
            "description": "对比等额本金和等额本息两种还款方式的核心差异，包括总利息、每月还款额、优缺点等（满足用户核心对比需求）",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "integer",
                        "description": "贷款金额（元），例如：1000000"
                    },
                    "years": {
                        "type": "integer",
                        "description": "贷款年限（年），例如：20"
                    },
                    "age": {
                        "type": "integer",
                        "description": "借款人年龄（岁），例如：30"
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "贷款年利率（%），默认3.85%，可选参数"
                    }
                },
                "required": ["loan_amount", "years", "age"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================
# 工具函数映射
available_functions = {
    "get_loan_repayment_methods": get_loan_repayment_methods,
    "get_repayment_detail": get_repayment_detail,
    "calculate_equal_principal": calculate_equal_principal,
    "calculate_equal_installment": calculate_equal_installment,
    "compare_repayment_methods": compare_repayment_methods
}

def run_loan_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行银行贷款Agent，处理用户查询（复刻原保险Agent逻辑）

    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端（对接阿里云通义千问，和原代码一致）
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的银行贷款顾问助手。你可以：
1. 介绍各种银行贷款还款方式及其详细信息
2. 计算等额本金、等额本息的还款额和总利息
3. 对比两种还款方式的核心差异并给出专业建议

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)

    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto"  # 让模型自主决定是否调用工具
            )
        except Exception as e:
            print(f"\n【错误】调用模型失败：{str(e)}")
            print("提示：请确保已设置环境变量 DASHSCOPE_API_KEY，且API密钥有效")
            return "抱歉，处理您的请求时遇到了模型调用错误。"

        response_message = response.choices[0].message

        # 将模型响应加入对话历史
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False, indent=2)}")

            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                # 截断过长的输出，保持整洁
                if len(function_response) > 500:
                    print(f"工具返回: {function_response[:500]}...（内容过长，已截断）")
                else:
                    print(f"工具返回: {function_response}")

                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                error_msg = f"未找到工具 {function_name}"
                print(f"错误：{error_msg}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps({"error": error_msg}, ensure_ascii=False)
                })

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"

# ==================== 示例运行（核心：你的需求场景） ====================
if __name__ == "__main__":
    # 运行你的核心需求：对比100万、20年、30岁的等额本金和等额本息
    user_query = "帮我比较一下等额本金和等额本息，贷款都是100万，我30岁，贷20年"
    run_loan_agent(user_query)
