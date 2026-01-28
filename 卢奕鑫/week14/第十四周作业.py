"""
智能旅行规划Agent - 完整可运行版本
使用DashScope API（兼容OpenAI格式）
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import random

# 如果没有安装dashscope，可以使用openai兼容模式
# pip install openai
try:
    from openai import OpenAI
    USE_OPENAI_CLIENT = True
except ImportError:
    # 如果openai不可用，使用模拟模式
    USE_OPENAI_CLIENT = False
    print("⚠️ 未安装openai库，使用模拟模式")


# ==================== 工具函数定义 ====================

class TravelTools:
    """旅行工具集合"""

    # 城市数据库
    CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "西安", "厦门"]

    # 航班数据库（模拟）
    FLIGHTS_DB = {
        "北京-上海": [
            {"airline": "中国国航", "flight_no": "CA1501", "departure": "08:00",
             "arrival": "10:15", "price": 850, "duration": "2h15m"},
            {"airline": "东方航空", "flight_no": "MU5101", "departure": "14:30",
             "arrival": "16:45", "price": 920, "duration": "2h15m"}
        ],
        "北京-广州": [
            {"airline": "南方航空", "flight_no": "CZ3101", "departure": "09:20",
             "arrival": "12:30", "price": 1100, "duration": "3h10m"}
        ],
        "上海-北京": [
            {"airline": "东方航空", "flight_no": "MU5115", "departure": "18:00",
             "arrival": "20:15", "price": 880, "duration": "2h15m"}
        ]
    }

    # 酒店数据库（模拟）
    HOTELS_DB = {
        "北京": [
            {"name": "北京王府井大酒店", "rating": 4.5, "price": 680,
             "location": "王府井", "types": ["标准间", "豪华间"]},
            {"name": "北京国贸饭店", "rating": 4.8, "price": 1200,
             "location": "国贸", "types": ["豪华间", "套房"]}
        ],
        "上海": [
            {"name": "上海外滩华尔道夫酒店", "rating": 4.9, "price": 1500,
             "location": "外滩", "types": ["豪华间", "套房"]},
            {"name": "上海静安洲际酒店", "rating": 4.6, "price": 950,
             "location": "静安区", "types": ["标准间", "豪华间"]}
        ],
        "广州": [
            {"name": "广州白天鹅宾馆", "rating": 4.7, "price": 850,
             "location": "沙面", "types": ["标准间", "豪华间", "套房"]}
        ]
    }

    # 景点数据库（模拟）
    ATTRACTIONS_DB = {
        "北京": [
            {"name": "故宫博物院", "category": "历史", "rating": 4.9,
             "ticket": 60, "duration": "4-5小时", "tags": ["世界文化遗产", "必去景点"]},
            {"name": "颐和园", "category": "历史", "rating": 4.8,
             "ticket": 30, "duration": "3-4小时", "tags": ["皇家园林", "风景优美"]},
            {"name": "长城八达岭", "category": "历史", "rating": 4.7,
             "ticket": 45, "duration": "5-6小时", "tags": ["世界奇迹", "户外活动"]}
        ],
        "上海": [
            {"name": "外滩", "category": "城市景观", "rating": 4.8,
             "ticket": 0, "duration": "2-3小时", "tags": ["夜景", "标志性建筑"]},
            {"name": "迪士尼乐园", "category": "娱乐", "rating": 4.9,
             "ticket": 399, "duration": "全天", "tags": ["主题公园", "亲子游"]}
        ],
        "杭州": [
            {"name": "西湖", "category": "自然", "rating": 4.9,
             "ticket": 0, "duration": "3-4小时", "tags": ["世界文化遗产", "浪漫"]},
            {"name": "灵隐寺", "category": "历史", "rating": 4.7,
             "ticket": 45, "duration": "2-3小时", "tags": ["千年古刹", "佛教圣地"]}
        ]
    }

    @staticmethod
    def search_flights(departure_city: str, arrival_city: str,
                      departure_date: str, return_date: Optional[str] = None) -> str:
        """搜索航班信息"""
        route = f"{departure_city}-{arrival_city}"

        result = {
            "query": {
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "departure_date": departure_date,
                "return_date": return_date
            },
            "departure_flights": [],
            "return_flights": []
        }

        # 查找去程航班
        if route in TravelTools.FLIGHTS_DB:
            for flight in TravelTools.FLIGHTS_DB[route]:
                flight_copy = flight.copy()
                flight_copy["date"] = departure_date
                result["departure_flights"].append(flight_copy)

        # 查找返程航班
        if return_date:
            return_route = f"{arrival_city}-{departure_city}"
            if return_route in TravelTools.FLIGHTS_DB:
                for flight in TravelTools.FLIGHTS_DB[return_route]:
                    flight_copy = flight.copy()
                    flight_copy["date"] = return_date
                    result["return_flights"].append(flight_copy)

        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def search_hotels(city: str, check_in_date: str, check_out_date: str,
                     guests: int = 2, room_type: str = "标准间") -> str:
        """搜索酒店信息"""

        # 计算住宿天数
        check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
        check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
        nights = (check_out - check_in).days

        result = {
            "query": {
                "city": city,
                "check_in": check_in_date,
                "check_out": check_out_date,
                "nights": nights,
                "guests": guests,
                "room_type": room_type
            },
            "hotels": []
        }

        if city in TravelTools.HOTELS_DB:
            for hotel in TravelTools.HOTELS_DB[city]:
                if room_type in hotel["types"]:
                    hotel_info = hotel.copy()
                    hotel_info["total_price"] = hotel["price"] * nights
                    hotel_info["price_per_night"] = hotel["price"]
                    result["hotels"].append(hotel_info)

        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def get_attractions(city: str, category: Optional[str] = None,
                       max_results: int = 5) -> str:
        """获取旅游景点"""
        result = {
            "city": city,
            "category": category,
            "attractions": []
        }

        if city in TravelTools.ATTRACTIONS_DB:
            for attraction in TravelTools.ATTRACTIONS_DB[city]:
                if not category or attraction["category"] == category:
                    result["attractions"].append(attraction)
                    if len(result["attractions"]) >= max_results:
                        break

        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def get_weather(city: str, date: str) -> str:
        """获取天气预报"""
        # 模拟天气数据
        conditions = ["晴", "多云", "小雨", "阴天", "雷阵雨"]
        temp_min = random.randint(15, 25)
        temp_max = random.randint(temp_min + 5, temp_min + 10)

        weather = {
            "city": city,
            "date": date,
            "temperature": f"{temp_min}-{temp_max}°C",
            "condition": random.choice(conditions),
            "humidity": f"{random.randint(40, 80)}%",
            "wind": f"{random.randint(1, 5)}级",
            "advice": TravelTools._get_weather_advice(random.choice(conditions))
        }

        return json.dumps(weather, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_weather_advice(condition: str) -> str:
        """根据天气给出建议"""
        advice_map = {
            "晴": "天气晴朗，适合户外活动",
            "多云": "天气舒适，适宜出行",
            "小雨": "建议携带雨具",
            "阴天": "天气较凉，建议添衣",
            "雷阵雨": "建议室内活动，注意安全"
        }
        return advice_map.get(condition, "天气多变，请注意")

    @staticmethod
    def create_itinerary(destination: str, days: int,
                        interests: List[str], budget: str = "中等") -> str:
        """创建旅行行程"""

        # 预算等级
        budget_levels = {
            "经济": {"daily": 300, "hotel": "经济型", "food": "快餐/小吃"},
            "中等": {"daily": 600, "hotel": "舒适型", "food": "餐厅用餐"},
            "豪华": {"daily": 1200, "hotel": "豪华型", "food": "高级餐厅"}
        }

        budget_info = budget_levels.get(budget, budget_levels["中等"])

        # 生成每日行程
        itinerary = []
        for day in range(1, days + 1):
            day_plan = {
                "day": day,
                "morning": TravelTools._generate_morning_activity(interests),
                "afternoon": TravelTools._generate_afternoon_activity(interests),
                "evening": TravelTools._generate_evening_activity(interests),
                "budget": budget_info["daily"]
            }
            itinerary.append(day_plan)

        result = {
            "destination": destination,
            "days": days,
            "interests": interests,
            "budget_level": budget,
            "total_budget": budget_info["daily"] * days,
            "recommendations": {
                "accommodation": budget_info["hotel"],
                "food": budget_info["food"],
                "transportation": "公共交通/打车"
            },
            "itinerary": itinerary
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _generate_morning_activity(interests: List[str]) -> str:
        """生成上午活动"""
        activities = {
            "历史": "参观历史遗迹或博物馆",
            "自然": "游览自然公园或风景区",
            "美食": "品尝当地特色早餐",
            "购物": "逛当地市场或购物中心",
            "娱乐": "参观主题公园"
        }

        for interest in interests:
            if interest in activities:
                return activities[interest]
        return "城市观光"

    @staticmethod
    def _generate_afternoon_activity(interests: List[str]) -> str:
        """生成下午活动"""
        activities = {
            "历史": "继续探索历史文化景点",
            "自然": "进行户外活动或徒步",
            "美食": "参加美食体验或烹饪课",
            "购物": "继续购物或寻找特色商品",
            "娱乐": "体验当地娱乐活动"
        }

        for interest in interests:
            if interest in activities:
                return activities[interest]
        return "自由活动"

    @staticmethod
    def _generate_evening_activity(interests: List[str]) -> str:
        """生成晚上活动"""
        activities = {
            "历史": "观看历史主题表演",
            "自然": "欣赏夜景或星空",
            "美食": "享受当地特色晚餐",
            "购物": "逛夜市",
            "娱乐": "观看演出或电影"
        }

        for interest in interests:
            if interest in activities:
                return activities[interest]
        return "当地文化体验"


# ==================== 工具函数映射 ====================

available_functions = {
    "search_flights": TravelTools.search_flights,
    "search_hotels": TravelTools.search_hotels,
    "get_attractions": TravelTools.get_attractions,
    "get_weather": TravelTools.get_weather,
    "create_itinerary": TravelTools.create_itinerary
}

# ==================== 工具Schema定义 ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索航班信息，包括航空公司、航班号、时间、价格等",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure_city": {"type": "string", "description": "出发城市"},
                    "arrival_city": {"type": "string", "description": "到达城市"},
                    "departure_date": {"type": "string", "description": "出发日期，格式：YYYY-MM-DD"},
                    "return_date": {"type": "string", "description": "返回日期，格式：YYYY-MM-DD"}
                },
                "required": ["departure_city", "arrival_city", "departure_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "搜索酒店信息，包括价格、评分、位置、房型等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "check_in_date": {"type": "string", "description": "入住日期，格式：YYYY-MM-DD"},
                    "check_out_date": {"type": "string", "description": "退房日期，格式：YYYY-MM-DD"},
                    "guests": {"type": "integer", "description": "入住人数"},
                    "room_type": {"type": "string", "description": "房型，如：标准间、豪华间、套房"}
                },
                "required": ["city", "check_in_date", "check_out_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_attractions",
            "description": "获取旅游景点信息，包括景点名称、类别、门票价格、游玩时间等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "category": {"type": "string", "description": "景点类别，如：历史、自然、娱乐等"},
                    "max_results": {"type": "integer", "description": "最大返回数量"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气预报，包括温度、天气状况、湿度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "查询日期，格式：YYYY-MM-DD"}
                },
                "required": ["city", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_itinerary",
            "description": "创建旅行行程规划，包括每日活动安排、预算建议等",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地城市"},
                    "days": {"type": "integer", "description": "旅行天数"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "兴趣列表，如：['历史', '美食', '购物']"
                    },
                    "budget": {"type": "string", "description": "预算等级：经济、中等、豪华"}
                },
                "required": ["destination", "days", "interests"]
            }
        }
    }
]

# ==================== Agent核心类 ====================

class TravelAgent:
    """智能旅行规划Agent"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.conversation_history = []

        # 系统提示
        system_prompt = """你是一位专业的旅行规划助手。你可以帮助用户：
1. 搜索航班信息
2. 搜索酒店住宿
3. 推荐旅游景点
4. 查询天气预报
5. 制定旅行行程

请根据用户的问题，使用合适的工具来获取信息，然后给出专业、友好的建议。
如果用户的问题不明确，请主动询问更多细节（如时间、预算、兴趣等）。"""

        self.conversation_history.append({
            "role": "system",
            "content": system_prompt
        })

        # 初始化API客户端
        self.client = None
        if USE_OPENAI_CLIENT and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                print("✅ 已初始化API客户端")
            except Exception as e:
                print(f"⚠️ API客户端初始化失败: {e}")
                self.client = None

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({"role": role, "content": content})

    def process_query(self, user_query: str) -> str:
        """处理用户查询"""
        print(f"\n👤 用户: {user_query}")

        # 添加用户消息
        self.add_message("user", user_query)

        # 如果没有API客户端，使用模拟模式
        if not self.client:
            return self._simulate_response(user_query)

        try:
            # 调用大模型
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # 添加助手响应到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls
            })

            # 检查是否需要工具调用
            if message.tool_calls:
                return self._handle_tool_calls(message.tool_calls)
            else:
                return message.content or "抱歉，我无法回答这个问题。"

        except Exception as e:
            print(f"⚠️ API调用错误: {e}")
            return self._simulate_response(user_query)

    def _handle_tool_calls(self, tool_calls) -> str:
        """处理工具调用"""
        tool_responses = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name

            print(f"\n🛠️ 调用工具: {function_name}")

            if function_name in available_functions:
                try:
                    # 解析参数
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"   参数: {json.dumps(function_args, ensure_ascii=False)}")

                    # 调用工具函数
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"   结果: 获取到{len(function_response)}字符的数据")

                    # 添加到工具响应列表
                    tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })

                    # 添加到对话历史
                    self.conversation_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })

                except Exception as e:
                    print(f"   错误: {e}")
                    error_response = json.dumps({"error": str(e)}, ensure_ascii=False)
                    tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": error_response
                    })
            else:
                print(f"   错误: 未知工具 {function_name}")

        # 如果有工具调用，需要再次调用大模型来处理结果
        if tool_responses:
            return self._process_tool_results()

        return "工具调用完成。"

    def _process_tool_results(self) -> str:
        """处理工具结果并生成最终回复"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=self.conversation_history
            )

            final_message = response.choices[0].message.content
            self.add_message("assistant", final_message)

            return final_message

        except Exception as e:
            print(f"⚠️ 处理工具结果时出错: {e}")
            return "已获取相关信息，请查看工具返回的结果。"

    def _simulate_response(self, user_query: str) -> str:
        """模拟响应（当没有API时使用）"""
        print("📱 使用模拟模式...")

        # 简单的关键词匹配
        query_lower = user_query.lower()

        if any(word in query_lower for word in ["航班", "飞机", "飞"]):
            # 模拟航班搜索
            result = TravelTools.search_flights("北京", "上海", "2024-05-01")
            return f"以下是航班信息：\n{result}\n\n需要我帮您预订吗？"

        elif any(word in query_lower for word in ["酒店", "住宿", "住"]):
            # 模拟酒店搜索
            result = TravelTools.search_hotels("北京", "2024-05-01", "2024-05-03")
            return f"以下是酒店信息：\n{result}"

        elif any(word in query_lower for word in ["景点", "玩", "旅游"]):
            # 模拟景点查询
            result = TravelTools.get_attractions("北京", "历史")
            return f"以下是景点推荐：\n{result}"

        elif "天气" in query_lower:
            # 模拟天气查询
            result = TravelTools.get_weather("北京", "2024-05-01")
            return f"天气预报：\n{result}"

        elif any(word in query_lower for word in ["行程", "规划", "安排"]):
            # 模拟行程规划
            result = TravelTools.create_itinerary("杭州", 3, ["自然", "美食"], "中等")
            return f"行程规划：\n{result}"

        else:
            return "我是旅行规划助手，可以帮您：\n1. 搜索航班和酒店\n2. 推荐景点\n3. 查询天气\n4. 制定行程\n\n请告诉我您的具体需求！"

    def run_demo(self):
        """运行演示"""
        print("\n" + "="*60)
        print("智能旅行规划Agent演示")
        print("="*60)

        demo_queries = [
            "我想查询从北京到上海的航班，5月1日出发",
            "帮我找一下北京的酒店，5月1日到3日，2个人",
            "推荐一些北京的历史景点",
            "查询北京5月1日的天气",
            "帮我规划一个3天的杭州行程，喜欢自然和美食，预算中等"
        ]

        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*40}")
            print(f"示例 {i}: {query}")
            print(f"{'='*40}")

            response = self.process_query(query)
            print(f"\n🤖 助手: {response}")

            # 简单解析JSON并展示
            if "{" in response and "}" in response:
                try:
                    # 提取JSON部分
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)

                        # 格式化显示
                        print(f"\n📊 解析结果:")
                        self._pretty_print_data(data)
                except:
                    pass

            input("\n按Enter继续...")  # 暂停

    def _pretty_print_data(self, data: Any, indent: int = 0):
        """美化打印数据"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print("  " * indent + f"📌 {key}:")
                    self._pretty_print_data(value, indent + 1)
                else:
                    print("  " * indent + f"  {key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data, 1):
                print("  " * indent + f"{i}.")
                self._pretty_print_data(item, indent + 1)
        else:
            print("  " * indent + str(data))

# ==================== 快速测试工具函数 ====================

def test_tools():
    """测试所有工具函数"""
    print("🧪 测试工具函数...")

    print("\n1. 测试航班搜索:")
    flights = TravelTools.search_flights("北京", "上海", "2024-05-01", "2024-05-03")
    print(flights)

    print("\n2. 测试酒店搜索:")
    hotels = TravelTools.search_hotels("北京", "2024-05-01", "2024-05-03", 2, "标准间")
    print(hotels)

    print("\n3. 测试景点查询:")
    attractions = TravelTools.get_attractions("北京", "历史", 3)
    print(attractions)

    print("\n4. 测试天气查询:")
    weather = TravelTools.get_weather("北京", "2024-05-01")
    print(weather)

    print("\n5. 测试行程规划:")
    itinerary = TravelTools.create_itinerary("杭州", 3, ["自然", "美食"], "中等")
    print(itinerary)

# ==================== 主程序 ====================

if __name__ == "__main__":

    print("🌍 智能旅行规划Agent")
    print("版本: 1.0")
    print("-" * 40)

    # 测试工具函数
    test_tools()

    print("\n" + "="*60)
    print("准备启动Agent...")
    print("="*60)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and USE_OPENAI_CLIENT:
        print("⚠️ 未找到API密钥，将使用模拟模式")
        print("请设置环境变量: export DASHSCOPE_API_KEY='your-key'")
        print("或直接在代码中设置")

    # 创建Agent
    agent = TravelAgent(api_key=api_key)

    # 运行演示
    agent.run_demo()

    # 交互模式
    print("\n" + "🌟" * 30)
    print("交互模式开始")
    print("输入'退出'或'quit'结束")
    print("🌟" * 30)

    while True:
        try:
            user_input = input("\n👤 您: ").strip()

            if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                print("👋 感谢使用，再见！")
                break

            if not user_input:
                continue

            response = agent.process_query(user_input)
            print(f"\n🤖 助手: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 已退出")
            break
        except Exception as e:
            print(f"\n⚠️ 错误: {e}")
