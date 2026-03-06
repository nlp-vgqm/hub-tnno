import json
import re
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from difflib import SequenceMatcher


# ---------------------- 【新增】重听指令检测函数 ----------------------
def is_rehear_command(user_input: str) -> bool:
    """
    检测用户输入是否为「重听指令」
    :param user_input: 用户输入文本
    :return: True=是重听指令，False=不是
    """
    # 支持的重听关键词（可按需扩展）
    rehear_keywords = ["重听", "再说一遍", "重复", "再说一次", "再说下", "重复一遍"]
    user_input = user_input.strip()
    return any(keyword in user_input for keyword in rehear_keywords)


# ---------------------- 工具函数（原有，未修改） ----------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离（Levenshtein Distance）"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def similarity(s1: str, s2: str) -> float:
    """计算文本相似度（基于编辑距离归一化）"""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def split_values(value_str: str) -> List[str]:
    """解析Excel中的槽位枚举值（逗号分隔）"""
    if pd.isna(value_str) or value_str.strip() == "":
        return []
    return [v.strip() for v in value_str.split(",")]


def read_excel_rows(file_path: str, sheet_name: str = 0) -> List[Dict]:
    """读取Excel为字典列表"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.fillna("")
    return df.to_dict("records")


# ---------------------- 数据结构（修改DialogueState） ----------------------
class Node:
    """对话节点"""

    def __init__(self, node_id: str, intent: str, slots: List[str], response: str, children: List[str] = None):
        self.node_id = node_id
        self.intent = intent
        self.slots = slots
        self.response = response
        self.children = children or []


class DialogueState:
    """对话状态跟踪"""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name  # 场景名
        self.current_node_id = None  # 当前节点ID
        self.filled_slots = {}  # 已填充槽位 {槽位名: 值}
        self.missing_slots = []  # 缺失槽位列表
        self.is_ended = False  # 对话是否结束
        # 【新增】存储上一次机器人回复
        self.last_response = None

    def reset(self):
        """重置对话状态（保留场景名）"""
        self.current_node_id = None
        self.filled_slots = {}
        self.missing_slots = []
        self.is_ended = False
        # 【新增】重置上一次回复
        self.last_response = None


class Scenario:
    """场景加载与管理"""

    def __init__(self, scenario_file: str):
        self.nodes: Dict[str, Node] = {}
        self.load_scenario(scenario_file)

    def load_scenario(self, file_path: str):
        """加载场景JSON文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenario_name = data.get("name")
        for node_data in data.get("nodes", []):
            node_id = f"{scenario_name}_{node_data['id']}"  # 场景前缀避免冲突
            node = Node(
                node_id=node_id,
                intent=node_data.get("intent", ""),
                slots=node_data.get("slots", []),
                response=node_data.get("response", ""),
                children=[f"{scenario_name}_{c}" for c in node_data.get("children", [])]
            )
            self.nodes[node_id] = node


class SlotOntology:
    """槽位本体（加载Excel中的槽位规则）"""

    def __init__(self, excel_path: str):
        self.slot_info: Dict[str, Dict] = {}
        self.load_slot_templates(excel_path)

    def load_slot_templates(self, file_path: str):
        """加载槽位模板Excel"""
        rows = read_excel_rows(file_path)
        for row in rows:
            slot_name = row.get("slot_name", "").strip()
            if not slot_name:
                continue
            self.slot_info[slot_name] = {
                "query": row.get("query", ""),  # 追问话术
                "values": split_values(row.get("values", "")),  # 枚举值
                "regex": row.get("regex", "")  # 正则匹配规则
            }


# ---------------------- 核心模块（NLU/DST/PM/NLG） ----------------------
class NLU:
    """自然语言理解（意图识别+槽位提取）"""

    def __init__(self, slot_ontology: SlotOntology):
        self.slot_ontology = slot_ontology

    def recognize_intent(self, user_input: str, nodes: Dict[str, Node], threshold: float = 0.6) -> Optional[Node]:
        """意图识别（基于相似度匹配）"""
        max_sim = 0.0
        best_node = None
        for node in nodes.values():
            sim = similarity(user_input, node.intent)
            if sim > max_sim and sim >= threshold:
                max_sim = sim
                best_node = node
        return best_node

    def extract_slots(self, user_input: str, required_slots: List[str]) -> Dict[str, str]:
        """槽位提取（枚举匹配+正则+自由文本）"""
        filled_slots = {}
        for slot in required_slots:
            slot_config = self.slot_ontology.slot_info.get(slot, {})
            # 1. 枚举值匹配
            for value in slot_config.get("values", []):
                if value in user_input:
                    filled_slots[slot] = value
                    break
            if slot in filled_slots:
                continue
            # 2. 正则匹配
            regex = slot_config.get("regex", "")
            if regex:
                match = re.search(regex, user_input)
                if match:
                    filled_slots[slot] = match.group(0)
                    continue
            # 3. 自由文本（取关键词，简化版）
            # 此处可扩展：实体识别/关键词提取等
            filled_slots[slot] = ""
        return filled_slots

    def recognize(self, user_input: str, state: DialogueState, scenario_nodes: Dict[str, Node]) -> Tuple[
        Optional[Node], Dict[str, str]]:
        """完整NLU流程：意图识别+槽位提取"""
        node = self.recognize_intent(user_input, scenario_nodes)
        slots = {}
        if node:
            slots = self.extract_slots(user_input, node.slots)
        return node, slots


class DST:
    """对话状态跟踪"""

    def update_state(self, state: DialogueState, nlu_result: Tuple[Optional[Node], Dict[str, str]]):
        """更新对话状态（填充槽位+检查缺失）"""
        node, extracted_slots = nlu_result
        if not node:
            state.missing_slots = []
            return
        # 更新当前节点
        state.current_node_id = node.node_id
        # 填充槽位
        for slot, value in extracted_slots.items():
            if value:
                state.filled_slots[slot] = value
        # 检查缺失槽位
        state.missing_slots = [s for s in node.slots if s not in state.filled_slots or not state.filled_slots[s]]
        # 检查对话是否结束（无缺失槽位且无子节点）
        if not state.missing_slots and not node.children:
            state.is_ended = True


class PM:
    """策略管理器（决策下一步动作）"""

    def get_policy(self, state: DialogueState) -> Dict:
        """生成策略：追问/回复/结束"""
        if not state.current_node_id:
            return {"action": "unknown", "message": "我没理解你的需求，能再说清楚点吗？"}
        if state.missing_slots:
            # 追问第一个缺失槽位
            slot = state.missing_slots[0]
            query = state.slot_ontology.slot_info.get(slot, {}).get("query", f"请问你{slot}是？")
            return {"action": "request_slot", "slot": slot, "message": query}
        if state.is_ended:
            return {"action": "end", "message": state.last_response or "任务完成！"}
        # 回复
        node = state.scenario_nodes.get(state.current_node_id)
        response = node.response if node else "已确认你的需求！"
        # 填充槽位到回复中
        for slot, value in state.filled_slots.items():
            response = response.replace(f"{{{slot}}}", value)
        return {"action": "reply", "message": response}


class NLG:
    """自然语言生成"""

    def generate(self, policy: Dict, state: DialogueState) -> str:
        """根据策略生成回复"""
        return policy.get("message", "抱歉，我有点没明白～")


# ---------------------- 核心对话系统（修改chat方法） ----------------------
class DialogueSystem:
    def __init__(self, scenarios_dir: str = "scenarios", slot_excel: str = "slot_templates.xlsx"):
        # 加载场景
        self.scenarios: Dict[str, Scenario] = {}
        self.load_scenarios(scenarios_dir)
        # 加载槽位本体
        self.slot_ontology = SlotOntology(slot_excel)
        # 初始化核心模块
        self.nlu = NLU(self.slot_ontology)
        self.dst = DST()
        self.pm = PM()
        self.nlg = NLG()
        # 会话管理
        self.sessions: Dict[str, DialogueState] = {}

    def load_scenarios(self, dir_path: str):
        """加载目录下所有场景文件"""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"场景目录不存在：{dir_path}")
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                scenario_name = filename[:-5]  # 去掉.json后缀
                file_path = os.path.join(dir_path, filename)
                self.scenarios[scenario_name] = Scenario(file_path)

    def list_scenarios(self) -> List[str]:
        """列出所有可用场景"""
        return list(self.scenarios.keys())

    def start(self, scenario_name: str, session_id: str) -> str:
        """启动新会话"""
        if scenario_name not in self.scenarios:
            return f"场景不存在！可用场景：{', '.join(self.list_scenarios())}"
        # 初始化对话状态
        self.sessions[session_id] = DialogueState(scenario_name)
        # 绑定场景节点和槽位本体到状态（方便后续调用）
        state = self.sessions[session_id]
        state.scenario_nodes = self.scenarios[scenario_name].nodes
        state.slot_ontology = self.slot_ontology
        return f"会话已启动，你可以开始咨询{scenario_name}相关需求～"

    def chat(self, session_id: str, user_input: str) -> Tuple[str, bool]:
        """
        处理用户输入，返回回复和对话是否结束
        :param session_id: 会话ID
        :param user_input: 用户输入文本
        :return: (回复文本, 是否结束)
        """
        # 1. 检查会话是否存在
        if session_id not in self.sessions:
            return "会话不存在，请先启动场景！", True

        state = self.sessions[session_id]
        user_input = user_input.strip()

        # 【新增】优先检测重听指令
        if is_rehear_command(user_input):
            if state.last_response:
                return state.last_response, state.is_ended
            else:
                return "还没有可重复的回复哦，你可以先说说你的需求～", state.is_ended

        # 2. 原有对话逻辑
        # NLU：意图识别+槽位提取
        nlu_result = self.nlu.recognize(user_input, state, state.scenario_nodes)
        # DST：更新对话状态
        self.dst.update_state(state, nlu_result)
        # PM：生成策略
        policy = self.pm.get_policy(state)
        # NLG：生成回复
        response = self.nlg.generate(policy, state)

        # 【新增】更新上一次回复
        state.last_response = response

        return response, state.is_ended


# ---------------------- 命令行测试入口 ----------------------
def main():
    # 初始化对话系统
    ds = DialogueSystem()
    print(f"可用场景：{', '.join(ds.list_scenarios())}")

    # 选择场景
    while True:
        scenario_name = input("请输入要启动的场景名（如buy_clothes）：").strip()
        if scenario_name in ds.list_scenarios():
            break
        print(f"场景不存在！可用场景：{', '.join(ds.list_scenarios())}")

    # 启动会话
    session_id = "test_session_001"
    start_msg = ds.start(scenario_name, session_id)
    print(start_msg)

    # 开始对话
    print("会话已启动，开始对话（输入q退出）：")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "q":
            print("退出对话")
            break
        response, is_ended = ds.chat(session_id, user_input)
        print(f"机器人：{response}")
        if is_ended:
            print("对话结束")
            break


if __name__ == "__main__":
    main()