'''
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        # 重听相关关键词 - 扩展更多可能的表达
        self.repeat_keywords = [
            "再说一遍", "重复一遍", "没听清", "没听清楚", "没听到",
            "重听", "再说一次", "什么", "啥", "啊", "嗯？", "pardon",
            "repeat", "again", "请重复", "能再说一遍吗", "刚才说什么",
            "没听见", "听不清", "可以重复吗"
        ]
        # 无效输入关键词（不会作为槽位值）
        self.invalid_inputs = [
            "没听清", "没听清楚", "没听到", "什么", "啥", "啊",
            "再说一遍", "重复一遍", "重听", "再说一次"
        ]

    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]


    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def check_repeat_request(self, query):
        """检查用户是否请求重听"""
        if not query:
            return False

        query = query.strip().lower()
        for keyword in self.repeat_keywords:
            if keyword in query:
                print(f"[重听检测] 检测到重听关键词: '{keyword}'")
                return True
        return False

    def is_valid_slot_value(self, query, slot_values, slot_name):
        """检查输入是否是有效的槽位值"""
        if not query:
            return False

        query = query.strip()

        # 如果是重听关键词，不是有效槽位值
        if self.check_repeat_request(query):
            print(f"[无效输入] '{query}' 是重听关键词，不作为槽位 '{slot_name}' 的值")
            return False

        # 如果是无效输入列表中的词，也不是有效槽位值
        for invalid in self.invalid_inputs:
            if invalid in query:
                print(f"[无效输入] '{query}' 包含无效关键词 '{invalid}'，不作为槽位值")
                return False

        # 检查是否匹配槽位值的正则表达式
        if isinstance(slot_values, str) and slot_values:
            try:
                # 对于时间槽位，特殊处理
                if slot_name == "时间":
                    # 简单的时间格式检查
                    time_patterns = [
                        r'今晚|明天|后天|今天',
                        r'\d+点|\d+:\d+',
                        r'上午|下午|晚上'
                    ]
                    for pattern in time_patterns:
                        if re.search(pattern, query):
                            print(f"[有效输入] '{query}' 匹配时间模式")
                            return True

                # 对于电影名称，特殊处理
                elif slot_name == "电影名称":
                    # 电影名称通常是中文词
                    if len(query) >= 2 and re.search(r'[\u4e00-\u9fff]+', query):
                        print(f"[有效输入] '{query}' 是有效电影名称")
                        return True

                # 通用正则匹配
                if re.search(slot_values, query):
                    print(f"[有效输入] '{query}' 匹配槽位模式 '{slot_values}'")
                    return True
                else:
                    print(f"[无效输入] '{query}' 不匹配槽位模式 '{slot_values}'")
            except re.error:
                if slot_values in query:
                    print(f"[有效输入] '{query}' 包含槽位值 '{slot_values}'")
                    return True
                else:
                    print(f"[无效输入] '{query}' 不包含槽位值 '{slot_values}'")
        return False

    def nlu(self, memory):
        # 先检查是否是重听请求
        if self.check_repeat_request(memory['query']):
            print(f"[NLU] 检测到重听请求")
            memory['is_repeat'] = True
            memory['skip_slot_filling'] = True
            # 重听请求不需要进行意图识别
            return memory

        memory['is_repeat'] = False
        memory['skip_slot_filling'] = False
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        #意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None

        # 如果没有可用节点，返回
        if not memory["available_nodes"]:
            memory["hit_node"] = None
            memory["intent_score"] = 0
            return memory

        for node in memory["available_nodes"]:
            if node in self.all_node_info:
                score = self.calucate_node_score(query, node)
                print(f"[意图评分] 节点 {node} 得分: {score}")
                if score > max_score:
                    max_score = score
                    hit_node = node

        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        print(f"[意图识别] 命中节点: {hit_node}, 得分: {max_score}")
        return memory


    def calucate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        if node not in self.all_node_info:
            return -1

        node_info = self.all_node_info[node]
        intent = node_info.get('intent', [])
        if not intent:
            return 0

        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        if not query or not sentence:
            return 0

        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        if len(union) == 0:
            return 0
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            return memory

        node_info = self.all_node_info[hit_node]
        query = memory['query']

        print(f"[槽位填充] 当前节点: {hit_node}, 需要填充的槽位: {node_info.get('slot', [])}")

        for slot in node_info.get('slot', []):
            # 如果槽位已经填充过，跳过
            if slot in memory:
                print(f"[槽位填充] 槽位 {slot} 已填充为 '{memory[slot]}'，跳过")
                continue

            if slot in self.slot_info:
                slot_values = self.slot_info[slot]["values"]
                print(f"[槽位填充] 尝试填充槽位 {slot}, 有效值模式: {slot_values}")

                # 检查是否是有效的槽位值
                if self.is_valid_slot_value(query, slot_values, slot):
                    try:
                        match = re.search(slot_values, query)
                        if match:
                            memory[slot] = match.group()
                            print(f"[槽位填充] ✓ 槽位 {slot} = '{match.group()}'")
                        else:
                            # 如果没有匹配到正则，但通过了验证，直接使用整个查询
                            memory[slot] = query
                            print(f"[槽位填充] ✓ 槽位 {slot} = '{query}'")
                    except re.error:
                        if slot_values in query:
                            memory[slot] = slot_values
                            print(f"[槽位填充] ✓ 槽位 {slot} = '{slot_values}'")
                        else:
                            memory[slot] = query
                            print(f"[槽位填充] ✓ 槽位 {slot} = '{query}'")
                else:
                    print(f"[槽位填充] ✗ 输入 '{query}' 不是槽位 {slot} 的有效值")

        # 打印当前已填充的所有槽位
        filled_slots = {k: v for k, v in memory.items() if k in node_info.get('slot', [])}
        print(f"[当前槽位状态] {filled_slots}")

        return memory

    def dst(self, memory):
        # 如果是重听请求，跳过dst
        if memory.get('is_repeat', False):
            print("[DST] 重听请求，跳过状态追踪")
            return memory

        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            memory["require_slot"] = None
            return memory

        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])

        # 检查是否有未填充的槽位
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                print(f"[DST] 需要填充槽位: {s}")
                return memory

        memory["require_slot"] = None
        print("[DST] 所有槽位已填充完成")
        return memory

    def dpo(self, memory):
        # 如果是重听请求，策略设为repeat
        if memory.get('is_repeat', False):
            memory["policy"] = "repeat"
            print("[DPO] 策略: repeat (重听)")
            return memory

        if memory.get("require_slot") is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["available_nodes"] = node_info.get("childnode", [])
                print(f"[DPO] 策略: reply, 后续节点: {memory['available_nodes']}")
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
            print(f"[DPO] 策略: request, 需要槽位: {memory['require_slot']}")
        return memory

    def nlg(self, memory):
        # 处理重听请求
        if memory.get('is_repeat', False):
            if 'last_response' in memory:
                memory["response"] = memory["last_response"]
                print(f"[NLG] 重听回复: {memory['response']}")
            else:
                memory["response"] = "抱歉，我还没有说过话呢。"
            # 重置重听标志
            memory['is_repeat'] = False
            return memory

        # 根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["response"] = self.fill_in_slot(node_info["response"], memory)
                print(f"[NLG] 回复: {memory['response']}")
            else:
                memory["response"] = "抱歉，我没有理解您的意思。"
        elif memory["policy"] == "request":
            slot = memory.get("require_slot")
            if slot and slot in self.slot_info:
                memory["response"] = self.slot_info[slot]["query"]
                print(f"[NLG] 请求: {memory['response']}")
            else:
                memory["response"] = "请提供更多信息。"

        # 保存本次回复作为历史记录
        memory["last_response"] = memory["response"]
        return memory

    def fill_in_slot(self, template, memory):
        """填充槽位到回复模板中"""
        result = template
        hit_node = memory.get("hit_node")
        if hit_node and hit_node in self.all_node_info:
            node_info = self.all_node_info[hit_node]
            for slot in node_info.get("slot", []):
                if slot in memory:
                    # 替换槽位标记
                    placeholder = f"#{slot}#"
                    # 确保只替换槽位标记，不替换其他内容
                    if placeholder in result:
                        result = result.replace(placeholder, memory[slot])
                        print(f"[模板填充] 替换 {placeholder} -> {memory[slot]}")
        return result

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        print(f"\n[用户输入] {query}")
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)

        # 返回结果，包括response和完整的memory
        return {
            'response': memory.get('response', ''),
            'memory': memory,
            'is_repeat': memory.get('is_repeat', False)
        }


# 网页接口函数 - 供前端调用
def process_message(user_input, session_memory=None):
    """
    处理用户消息的接口函数

    Args:
        user_input: 用户输入的文本
        session_memory: 会话记忆（可选）

    Returns:
        dict: 包含机器人回复和更新后的记忆
    """
    # 初始化对话系统
    ds = DialogueSystem()

    # 初始化或使用已有的记忆
    if session_memory is None:
        memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}
    else:
        memory = session_memory

    # 处理用户输入
    result = ds.run(user_input, memory)

    return {
        'response': result['response'],
        'memory': result['memory']
    }


if __name__ == '__main__':
    # 测试代码
    ds = DialogueSystem()
    print("=" * 60)
    print("任务型多轮对话系统 - 测试模式")
    print("=" * 60)
    print("你可以说'没听清'来重听上一句话")
    print("输入 'quit' 退出")
    print("=" * 60)

    memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}

    while True:
        user_input = input("\n用户：")
        if user_input.lower() == 'quit':
            print("机器人：再见！")
            break

        result = ds.run(user_input, memory)
        memory = result['memory']  # 更新记忆
        print(f"机器人：{result['response']}")
