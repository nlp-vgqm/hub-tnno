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

    def load(self):
        self.all_node_info = {}
        self.load_scenario(r"scenario-买衣服.json")
        self.load_scenario(r"scenario-看电影.json")
        self.load_slot_templet(r"slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in
                                                                                     node['childnode']]

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        # 三列：slot, query, values
        self.slot_info = {}
        # 逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        # 意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calucate_node_score(self, query, node):
        # 节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calucate_sentence_score(self, query, sentence):
        # 两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        # 槽位填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                match = re.search(slot_values, memory["query"])
                if match:
                    memory[slot] = match.group()
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            # 有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            # policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            template = template.replace(slot, memory[slot])
        return template

    def is_repeat_request(self, query):
        """检测是否为重听/重复请求"""
        repeat_keywords = [
            "重播", "再讲一遍", "听不清", "重复一下", "再说一遍",
            "请重复", "再发一遍", "重说", "再听一遍", "麻烦重复"
        ]
        for keyword in repeat_keywords:
            if keyword in query:
                return True
        return False

    def run(self, query, memory):
        '''
        query: 用戶输入
        memory: 用户状态
        '''
        # 1. 检查是否为重听请求
        if self.is_repeat_request(query):
            # 直接返回上一轮的 response，不走完整流程
            if "last_response" in memory and memory["last_response"]:
                memory["response"] = memory["last_response"]
                return memory
            else:
                # 如果没有历史响应，返回默认提示
                memory["response"] = "抱歉，我没有上一条信息可以重复。"
                return memory

        # 2. 正常流程
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)

        # 3. 保存本次 response 到 last_response，供下次重听使用
        memory["last_response"] = memory["response"]

        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.slot_info)

    # 初始状态
    memory = {
        "available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"],
        "last_response": ""  # 初始无上一次响应
    }

    while True:
        query = input("请输入：").strip()
        if not query:
            continue

        memory = ds.run(query, memory)
        print("系统输出：", memory["response"])
        print("===========")
