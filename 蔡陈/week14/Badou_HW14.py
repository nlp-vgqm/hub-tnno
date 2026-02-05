"""
留学顾问Agent示例 - Function Call + 半结构化Markdown最终输出（稳定版）

修复/增强点：
- messages 统一使用 dict（避免 ChatCompletionMessage 没有 .get 的报错）
- tool_calls 兼容 dict/对象两种返回
- 达到 max_iterations 也兜底输出 Markdown（不再丢最终结果）
- preference 支持 str/dict（修复 'str' object has no attribute 'get'）

注意：本脚本的院校/费用/要求是“示例数据”，用于演示 agent + tool call 流程。
"""

import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI

# ==================== deepseek 配置（按你提供的） ====================
DEEPSEEK_MODEL_NAME = "deepseek-v3.2"
DEEPSEEK_BASE_URL = "https://api.drqyq.com/v1"
DEEPSEEK_API_KEY = "sk-ZtcHN0Rbad31kuQGCFPFxmKDNuFB3iLG8sFtgZKXysByAC3o"

# DashScope（Qwen）默认 base_url（如果不用可忽略）
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ==================== 工具函数定义（示例数据/简化逻辑） ====================

def get_destination_options():
    """获取可选留学目的地概览（示例数据）"""
    options = [
        {
            "country": "美国",
            "strengths": ["科研与产业结合强", "专业选择广", "Top学校多"],
            "language": "英语",
            "typical_length": {"本科": "4年", "硕士": "1-2年", "博士": "4-6年"},
            "annual_cost_rmb_range": [450000, 900000],
            "notes": ["部分项目要求GRE/GMAT", "硕士/博士更看科研/实习"]
        },
        {
            "country": "英国",
            "strengths": ["硕士学制短", "申请节奏快", "课程型项目成熟"],
            "language": "英语",
            "typical_length": {"本科": "3年", "硕士": "1年", "博士": "3-4年"},
            "annual_cost_rmb_range": [350000, 650000],
            "notes": ["硕士多看本科GPA + 语言", "热门项目竞争激烈"]
        },
        {
            "country": "加拿大",
            "strengths": ["教育质量高", "性价比中等", "长期规划空间较大(视政策变化)"],
            "language": "英语/法语",
            "typical_length": {"本科": "4年", "硕士": "1-2年", "博士": "4-6年"},
            "annual_cost_rmb_range": [280000, 550000],
            "notes": ["硕博更看科研/导师匹配", "部分项目偏研究型"]
        },
        {
            "country": "澳大利亚",
            "strengths": ["学制灵活", "申请相对友好", "热门专业多"],
            "language": "英语",
            "typical_length": {"本科": "3-4年", "硕士": "1-2年", "博士": "3-5年"},
            "annual_cost_rmb_range": [280000, 600000],
            "notes": ["部分项目接受多种语言考试", "可配课/衔接"]
        },
        {
            "country": "德国",
            "strengths": ["公立学费低/免学费项目多", "工科强", "性价比高"],
            "language": "德语/英语(英授项目)",
            "typical_length": {"本科": "3-4年", "硕士": "2年", "博士": "3-5年"},
            "annual_cost_rmb_range": [120000, 300000],
            "notes": ["德语项目通常需德语证明", "英授也常要英语成绩"]
        },
        {
            "country": "日本",
            "strengths": ["整体费用可控", "理工方向强", "导师制研究路径清晰"],
            "language": "日语/英语(少量英授)",
            "typical_length": {"本科": "4年", "硕士": "2年", "博士": "3年"},
            "annual_cost_rmb_range": [120000, 280000],
            "notes": ["日语能力重要", "研究型强调导师匹配"]
        },
        {
            "country": "新加坡",
            "strengths": ["亚洲科研/就业枢纽", "英授为主", "学校集中度高"],
            "language": "英语",
            "typical_length": {"本科": "3-4年", "硕士": "1-2年", "博士": "4-5年"},
            "annual_cost_rmb_range": [250000, 550000],
            "notes": ["竞争激烈", "偏好科研/竞赛/实习亮点"]
        }
    ]
    return json.dumps(options, ensure_ascii=False)


def get_program_requirements(level: str, country: str, track: str):
    """获取某国家某层级典型申请要求模板（示例）"""
    req = {
        "硕士": {
            "英国": {
                "授课型": {
                    "academic": ["本科GPA/均分", "专业背景匹配", "PS/推荐信"],
                    "language": ["IELTS(常见6.5-7.0+，单项要求)", "TOEFL(部分)"],
                    "notes": ["学制短，节奏快，热门项目竞争激烈"]
                }
            },
            "美国": {
                "授课型": {
                    "academic": ["本科GPA", "相关专业背景", "简历/文书/推荐信"],
                    "tests": ["GRE/GMAT(视项目)", "部分CS/工程可不要求但有加分"],
                    "language": ["TOEFL/IELTS(视项目)"],
                    "notes": ["实习/科研经历很关键，择校分梯度更稳"]
                }
            }
        },
        "博士": {
            "美国": {
                "PhD": {
                    "academic": ["本科/硕士成绩", "科研经历(非常关键)", "推荐信(非常关键)"],
                    "tests": ["GRE(视项目)", "研究陈述/写作样本"],
                    "language": ["TOEFL/IELTS(视项目)"],
                    "notes": ["多提供TA/RA/funding，核心看导师/方向/成果"]
                }
            }
        }
    }

    level_map = req.get(level, {})
    country_map = level_map.get(country, {})
    track_map = country_map.get(track)

    if not track_map:
        return json.dumps({"error": "未找到匹配的要求模板", "hint": "请检查 level/country/track 是否在示例范围内"}, ensure_ascii=False)

    return json.dumps({
        "level": level,
        "country": country,
        "track": track,
        "requirements": track_map
    }, ensure_ascii=False)


def estimate_study_cost(country: str, level: str, years: int, city_tier: str, tuition_band: str):
    """估算留学费用（示例模型）"""
    living_cost_table = {"一线": 160000, "二线": 120000, "小城": 90000}
    base_tuition = {
        "美国": {"本科": 320000, "硕士": 340000, "博士": 280000},
        "英国": {"本科": 260000, "硕士": 300000, "博士": 220000},
        "加拿大": {"本科": 180000, "硕士": 220000, "博士": 180000},
        "澳大利亚": {"本科": 200000, "硕士": 240000, "博士": 200000},
        "德国": {"本科": 60000, "硕士": 80000, "博士": 60000},
        "日本": {"本科": 70000, "硕士": 80000, "博士": 80000},
        "新加坡": {"本科": 220000, "硕士": 260000, "博士": 220000}
    }
    band_factor = {"低": 0.85, "中": 1.0, "高": 1.25}

    if country not in base_tuition or level not in base_tuition[country]:
        return json.dumps({"error": "暂不支持该国家或层级的费用估算"}, ensure_ascii=False)
    if city_tier not in living_cost_table or tuition_band not in band_factor:
        return json.dumps({"error": "city_tier 或 tuition_band 参数不合法"}, ensure_ascii=False)

    tuition_per_year = base_tuition[country][level] * band_factor[tuition_band]
    living_per_year = living_cost_table[city_tier]

    misc_one_time = 35000
    misc_per_year = 15000

    annual_total = tuition_per_year + living_per_year + misc_per_year
    total = annual_total * years + misc_one_time

    return json.dumps({
        "country": country,
        "level": level,
        "years": years,
        "city_tier": city_tier,
        "tuition_band": tuition_band,
        "tuition_per_year_rmb": round(tuition_per_year, 0),
        "living_per_year_rmb": round(living_per_year, 0),
        "misc_one_time_rmb": misc_one_time,
        "misc_per_year_rmb": misc_per_year,
        "annual_total_rmb": round(annual_total, 0),
        "estimated_total_rmb": round(total, 0),
        "note": "示例估算：仅用于方案对比，实际以学校学费、城市消费、汇率与个人消费为准"
    }, ensure_ascii=False)


def recommend_countries_and_paths(profile: Dict[str, Any]):
    """
    推荐国家与路径（示例规则）
    修复：preference 支持 str/dict
    """
    level = profile.get("level")
    budget = profile.get("budget_per_year_rmb", 0)
    raw_pref = profile.get("preference", {}) or {}

    # preference 兼容：str -> dict
    if isinstance(raw_pref, str):
        s = raw_pref.lower()
        pref = {
            "duration_short": any(k in s for k in ["短", "别太长", "学制短", "想快", "快点"]),
            "immigration": any(k in s for k in ["移民", "想移民"]),
            "research": any(k in s for k in ["科研", "研究", "偏科研"]),
            "english_only": any(k in s for k in ["全英文", "只要英文", "english only"])
        }
    elif isinstance(raw_pref, dict):
        pref = raw_pref
    else:
        pref = {}

    candidates = []

    def add(country, path, score, reasons):
        candidates.append({
            "country": country,
            "suggested_path": path,
            "fit_score": score,
            "reasons": reasons
        })

    if pref.get("duration_short") and level in ("硕士", "博士"):
        add("英国", "授课型硕士（多为1年）+ 强调专业匹配", 86,
            ["学制短", "数据相关项目多", "符合‘学制别太长’"])

    add("新加坡", "头部院校集中申请 + 强化实习/项目经历", 78,
        ["英授为主", "就业/科研密度高", "竞争激烈"])

    if budget >= 500000:
        add("美国", "梯度选校：冲刺+匹配+保底，重视背景提升", 84,
            ["资源丰富", "数据/CS方向选择广", "实习机会多"])
    else:
        add("加拿大", "1.5-2年硕士（偏研究/实践）", 80,
            ["综合性价比", "路径稳健(视政策变化)", "项目类型多"])

    # 预算修正
    for c in candidates:
        if c["country"] == "美国" and budget < 400000:
            c["fit_score"] -= 8
            c["reasons"].append("预算偏紧：更适合奖学金/性价比项目或其他国家")
    candidates.sort(key=lambda x: x["fit_score"], reverse=True)

    return json.dumps({
        "input_profile": profile,
        "parsed_preference": pref,
        "recommendations": candidates[:5],
        "note": "示例推荐：建议结合具体院校项目要求与时间线进一步细化"
    }, ensure_ascii=False)


def recommend_schools_and_majors(country: str, level: str, major_interest: str, constraints: Dict[str, Any]):
    """择校与专业建议（示例清单）"""
    sample = {
        "英国": [
            {"school": "Sample UK University A", "major": f"{major_interest} MSc", "fit_tags": ["学制短", "课程紧凑"], "tier": "冲刺/匹配"},
            {"school": "Sample UK University B", "major": f"{major_interest} MSc", "fit_tags": ["项目实践多"], "tier": "匹配"},
            {"school": "Sample UK University C", "major": f"{major_interest} MSc", "fit_tags": ["性价比", "课程友好"], "tier": "保底/匹配"},
        ],
        "新加坡": [
            {"school": "Sample SG University", "major": f"{major_interest}", "fit_tags": ["国际化", "竞争激烈"], "tier": "冲刺/匹配"}
        ],
        "加拿大": [
            {"school": "Sample Canada University A", "major": f"{major_interest}", "fit_tags": ["科研环境好"], "tier": "匹配"},
            {"school": "Sample Canada University B", "major": f"{major_interest}", "fit_tags": ["性价比", "就业支持"], "tier": "匹配/保底"}
        ],
        "美国": [
            {"school": "Sample US University A", "major": f"{major_interest} (STEM Track)", "fit_tags": ["科研机会多", "产业合作"], "tier": "冲刺/匹配"},
            {"school": "Sample US University B", "major": f"{major_interest}", "fit_tags": ["就业导向"], "tier": "匹配"},
        ]
    }

    if country not in sample:
        return json.dumps({"error": "暂不支持该国家的院校示例"}, ensure_ascii=False)

    return json.dumps({
        "country": country,
        "level": level,
        "major_interest": major_interest,
        "constraints": constraints,
        "recommend_list": sample[country],
        "note": "示例学校清单：用于展示工具返回格式；实际请接入院校库/项目库"
    }, ensure_ascii=False)


# ==================== 工具 Schema（给模型） ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_destination_options",
            "description": "获取可选留学目的地（国家/地区）概览：语言环境、学制、费用区间、优势等",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_program_requirements",
            "description": "获取某国家某层级的典型申请要求模板（示例）",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string"},
                    "country": {"type": "string"},
                    "track": {"type": "string"}
                },
                "required": ["level", "country", "track"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_countries_and_paths",
            "description": "根据用户背景（成绩、语言、预算、偏好等）推荐国家与路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "object"}
                },
                "required": ["profile"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_study_cost",
            "description": "估算留学费用（示例模型）",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "level": {"type": "string"},
                    "years": {"type": "integer"},
                    "city_tier": {"type": "string"},
                    "tuition_band": {"type": "string"}
                },
                "required": ["country", "level", "years", "city_tier", "tuition_band"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_schools_and_majors",
            "description": "推荐学校与专业（示例清单）",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "level": {"type": "string"},
                    "major_interest": {"type": "string"},
                    "constraints": {"type": "object"}
                },
                "required": ["country", "level", "major_interest", "constraints"]
            }
        }
    }
]

available_functions = {
    "get_destination_options": get_destination_options,
    "get_program_requirements": get_program_requirements,
    "recommend_countries_and_paths": recommend_countries_and_paths,
    "estimate_study_cost": estimate_study_cost,
    "recommend_schools_and_majors": recommend_schools_and_majors
}


# ==================== Markdown 汇总（从工具结果生成最终答复） ====================

def _try_parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_tool_outputs(messages: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    outputs: Dict[str, List[Any]] = {}
    for m in messages:
        if m.get("role") == "tool":
            name = m.get("name", "tool")
            content = m.get("content", "")
            parsed = _try_parse_json(content)
            outputs.setdefault(name, []).append(parsed if parsed is not None else content)
    return outputs


def build_markdown_from_tools(user_query: str, messages: List[Dict[str, Any]]) -> str:
    tool_outputs = extract_tool_outputs(messages)

    # 最后一个 assistant 的 content（用于补充说明）
    model_text = None
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            model_text = m.get("content")
            break

    recs = None
    if "recommend_countries_and_paths" in tool_outputs and tool_outputs["recommend_countries_and_paths"]:
        last = tool_outputs["recommend_countries_and_paths"][-1]
        if isinstance(last, dict):
            recs = last

    costs = tool_outputs.get("estimate_study_cost", [])
    schools = tool_outputs.get("recommend_schools_and_majors", [])

    md: List[str] = []
    md.append("# 留学规划建议（半结构化）\n")

    md.append("## 用户输入")
    md.append(f"- {user_query}\n")

    md.append("## 背景摘要（自动提取）")
    if isinstance(recs, dict):
        profile = recs.get("input_profile", {}) or {}
        academics = profile.get("academics", {}) if isinstance(profile.get("academics"), dict) else {}
        language = profile.get("language", {})
        gpa = academics.get("gpa")
        toefl = language.get("toefl") if isinstance(language, dict) else language

        md.append(f"- 目标层级：{profile.get('level', '未提供')}")
        md.append(f"- 专业方向：{profile.get('major_interest', '未提供')}")
        md.append(f"- GPA：{gpa if gpa is not None else '未提供'}")
        md.append(f"- 语言（TOEFL/IELTS等）：{toefl if toefl is not None else '未提供'}")
        md.append(f"- 年预算（RMB）：{profile.get('budget_per_year_rmb', '未提供')}")
        md.append(f"- 偏好（解析）：{json.dumps(recs.get('parsed_preference', {}), ensure_ascii=False)}")
    else:
        md.append("- 未获取到结构化 profile（可能模型未调用 recommend_countries_and_paths）")

    md.append("\n---\n")

    md.append("## 推荐国家与路径（按适配度排序）")
    if isinstance(recs, dict) and recs.get("recommendations"):
        for r in recs["recommendations"]:
            md.append(f"### {r.get('country','未知')}（适配度：{r.get('fit_score','NA')}）")
            md.append(f"- 路径建议：{r.get('suggested_path','未提供')}")
            reasons = r.get("reasons", [])
            if isinstance(reasons, list) and reasons:
                md.append(f"- 理由：{'；'.join(reasons)}")
            md.append("")
    else:
        md.append("- 暂无国家推荐结果")

    md.append("\n---\n")

    md.append("## 费用估算（示例）")
    if costs:
        for c in costs:
            if isinstance(c, dict):
                md.append(f"### {c.get('country','未知')} · {c.get('level','')} · 学制 {c.get('years','?')} 年")
                md.append(f"- 年度总费用（估算）：**{c.get('annual_total_rmb','NA')} RMB**")
                md.append(f"- 项目总费用（估算）：**{c.get('estimated_total_rmb','NA')} RMB**")
                md.append(f"- 备注：{c.get('note','')}\n")
            else:
                md.append(f"- 原始费用输出：`{str(c)[:200]}`")
    else:
        md.append("- 本次未计算费用（如果你希望强制计算，可在代码里自动补算）")

    md.append("\n---\n")

    md.append("## 择校与专业建议（示例清单）")
    if schools:
        for s in schools:
            if isinstance(s, dict):
                md.append(f"### {s.get('country','')} · {s.get('level','')} · {s.get('major_interest','')}")
                rec_list = s.get("recommend_list", [])
                if isinstance(rec_list, list) and rec_list:
                    for item in rec_list:
                        md.append(f"- **{item.get('school','未知学校')}** — {item.get('major','')}"
                                  f"（{item.get('tier','')}）")
                        tags = item.get("fit_tags", [])
                        if isinstance(tags, list) and tags:
                            md.append(f"  - 标签：{', '.join(tags)}")
                    md.append("")
                else:
                    md.append("- （该工具未返回清单）\n")
    else:
        md.append("- 本次未返回择校清单")

    md.append("\n---\n")

    md.append("## 下一步建议")
    md.append("1. 选定 1–2 个主申国家 + 1 个备选国家（结合学制/预算/就业偏好）。")
    md.append("2. 每档（冲刺/匹配/保底）各选 2–4 个项目，形成申请组合。")
    md.append("3. 核对项目要求：先修课、语言门槛、是否需要 GRE/GMAT、截止日期。")
    md.append("4. 强化数据相关经历：项目、实习、科研、竞赛、作品集（GitHub/报告）。")

    if model_text:
        md.append("\n---\n")
        md.append("## 模型补充（自由文本）")
        md.append(model_text)

    return "\n".join(md)


# ==================== 工具调用解析（兼容 dict/对象） ====================

def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """把 ChatCompletionMessage(pydantic) 转成 dict；如果本来就是 dict 直接返回"""
    if isinstance(msg, dict):
        return msg
    # 兼容不同版本 openai SDK
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    # 最后兜底：尽量构造
    return {
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
        "tool_calls": getattr(msg, "tool_calls", None),
    }


def _tool_call_to_parts(tool_call: Any):
    """
    返回 (tool_call_id, function_name, arguments_str)
    tool_call 可能是 dict 或对象
    """
    if isinstance(tool_call, dict):
        tc_id = tool_call.get("id")
        fn = tool_call.get("function", {}) or {}
        fn_name = fn.get("name")
        args = fn.get("arguments") or "{}"
        return tc_id, fn_name, args

    # object style
    tc_id = getattr(tool_call, "id", None)
    fn_obj = getattr(tool_call, "function", None)
    fn_name = getattr(fn_obj, "name", None) if fn_obj else None
    args = getattr(fn_obj, "arguments", None) if fn_obj else None
    return tc_id, fn_name, args or "{}"


# ==================== Agent 核心逻辑（稳定输出 Markdown） ====================

def run_agent(
    user_query: str,
    api_key: str = None,
    model: str = DEEPSEEK_MODEL_NAME,
    base_url: str = None,
    markdown_final: bool = True,
    max_iterations: int = 5,
):
    # 选择 base_url/key
    chosen_base_url = base_url
    chosen_api_key = api_key

    if model == DEEPSEEK_MODEL_NAME:
        chosen_base_url = chosen_base_url or DEEPSEEK_BASE_URL
        chosen_api_key = chosen_api_key or DEEPSEEK_API_KEY

    if not chosen_api_key:
        chosen_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

    if not chosen_base_url:
        chosen_base_url = DASHSCOPE_BASE_URL

    client = OpenAI(api_key=chosen_api_key, base_url=chosen_base_url)

    # messages 永远存 dict
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": """你是一位专业的留学规划顾问助手。你可以：
1) 介绍不同国家/地区的留学特点（语言、学制、费用范围、优势）
2) 解释不同层级（本科/硕士/博士）的申请要求：语言/GPA/GRE/GMAT/科研等
3) 根据学生背景与偏好推荐国家与申请路径，并说明理由
4) 估算留学费用（年费用与总费用）
5) 给出学校与专业方向的匹配建议（示例清单）

要求：
- 优先调用工具获取结构化信息；
- 信息足够后停止调用工具，给出总结建议。
"""
        },
        {"role": "user", "content": user_query}
    ]

    print("\n" + "=" * 60)
    print("【用户问题】")
    print(user_query)
    print("=" * 60)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # 转 dict 后存入 messages
        response_message = response.choices[0].message
        response_message_dict = _message_to_dict(response_message)
        messages.append(response_message_dict)

        tool_calls = response_message_dict.get("tool_calls")

        # 如果模型不再调用工具：输出最终结果
        if not tool_calls:
            if markdown_final:
                final_md = build_markdown_from_tools(user_query, messages)
                print("\n【Agent最终（Markdown）回复】\n")
                print(final_md)
                print("\n" + "=" * 60)
                return final_md
            else:
                print("\n【Agent最终回复】")
                print(response_message_dict.get("content"))
                print("=" * 60)
                return response_message_dict.get("content")

        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")

        for tc in tool_calls:
            tool_call_id, function_name, arguments_str = _tool_call_to_parts(tc)

            # arguments 解析
            try:
                function_args = json.loads(arguments_str or "{}")
            except Exception:
                function_args = {}

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
            else:
                function_response = json.dumps({"error": f"未找到工具 {function_name}"}, ensure_ascii=False)

            if isinstance(function_response, str) and len(function_response) > 200:
                print(f"工具返回: {function_response[:200]}...")
            else:
                print(f"工具返回: {function_response}")

            # tool 结果以 dict 存入 messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": function_response
            })

    # 达到最大轮次：兜底输出 Markdown（关键）
    print("\n【警告】达到最大迭代次数，Agent循环结束")

    if markdown_final:
        final_md = build_markdown_from_tools(user_query, messages)
        print("\n【Agent最终（Markdown-兜底）回复】\n")
        print(final_md)
        print("\n" + "=" * 60)
        return final_md

    return "抱歉，处理您的请求时遇到了问题。"


# ==================== 示例运行 ====================
if __name__ == "__main__":
    run_agent(
        "我本科GPA 3.5，托福95，想申请硕士，专业偏数据科学，预算一年35万，想学制别太长，推荐国家和大概费用。",
        model=DEEPSEEK_MODEL_NAME,   # 用 deepseek-v3.2
        markdown_final=True,
        max_iterations=5
    )
