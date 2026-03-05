from flask import Flask, request, jsonify, render_template_string
import uuid
from task_dialogue_system import DialogueSystem

# 初始化Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持JSON中文显示

# 初始化对话系统
SLOT_EXCEL = "slot_templates.xlsx"
SCENARIOS_DIR = "scenarios"
ds = DialogueSystem(scenarios_dir=SCENARIOS_DIR, slot_excel=SLOT_EXCEL)

# 会话存储（内存版，仅用于演示）
SESSIONS = {}

# ---------------------- 前端页面模板 ----------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>任务型对话系统</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ccc; border-radius: 8px; padding: 20px; height: 500px; overflow-y: auto; margin-bottom: 20px; }
        .message { margin: 10px 0; padding: 8px 12px; border-radius: 4px; max-width: 70%; }
        .user-message { background-color: #007bff; color: white; margin-left: auto; }
        .bot-message { background-color: #e9ecef; color: black; margin-right: auto; }
        .control-panel { display: flex; gap: 10px; margin-bottom: 20px; }
        .control-panel select, .control-panel button { padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; }
        .input-panel { display: flex; gap: 10px; }
        .input-panel input { flex: 1; padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; }
        .input-panel button { padding: 8px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>任务型对话系统（支持重听功能）</h1>
    <div class="control-panel">
        <select id="scenario-select">
            {% for scenario in scenarios %}
            <option value="{{ scenario }}">{{ scenario }}</option>
            {% endfor %}
        </select>
        <button onclick="startSession()">启动会话</button>
        <span id="session-status">未启动</span>
    </div>
    <div class="chat-container" id="chat-container"></div>
    <div class="input-panel">
        <input type="text" id="user-input" placeholder="输入消息（支持“重听/再说一遍”等指令）" />
        <button onclick="sendMessage()">发送</button>
    </div>

    <script>
        let currentSessionId = null;

        // 获取场景列表
        async function loadScenarios() {
            const res = await fetch('/api/scenarios');
            const data = await res.json();
            const select = document.getElementById('scenario-select');
            select.innerHTML = data.scenarios.map(s => `<option value="${s}">${s}</option>`).join('');
        }

        // 启动会话
        async function startSession() {
            const scenario = document.getElementById('scenario-select').value;
            const res = await fetch('/api/session/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenario })
            });
            const data = await res.json();
            if (data.code === 0) {
                currentSessionId = data.session_id;
                document.getElementById('session-status').textContent = `会话已启动：${currentSessionId}`;
                addMessage('系统', data.message, 'bot');
                document.getElementById('user-input').disabled = false;
            } else {
                alert(data.message);
            }
        }

        // 发送消息
        async function sendMessage() {
            if (!currentSessionId) {
                alert('请先启动会话！');
                return;
            }
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;

            // 添加用户消息
            addMessage('你', text, 'user');
            input.value = '';

            // 调用接口获取回复
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: text
                })
            });
            const data = await res.json();
            // 添加机器人消息
            addMessage('机器人', data.message, 'bot');
            // 对话结束则禁用输入
            if (data.is_ended) {
                document.getElementById('session-status').textContent += '（已结束）';
                document.getElementById('user-input').disabled = true;
            }
        }

        // 添加消息到聊天窗口
        function addMessage(sender, content, type) {
            const container = document.getElementById('chat-container');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${type}-message`;
            msgDiv.innerHTML = `<strong>${sender}：</strong>${content}`;
            container.appendChild(msgDiv);
            // 滚动到底部
            container.scrollTop = container.scrollHeight;
        }

        // 页面加载时加载场景
        window.onload = loadScenarios;
    </script>
</body>
</html>
"""


# ---------------------- API接口 ----------------------
@app.route('/api/scenarios', methods=['GET'])
def list_scenarios():
    """获取可用场景列表"""
    return jsonify({
        "code": 0,
        "scenarios": ds.list_scenarios()
    })


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """启动新会话"""
    data = request.get_json()
    scenario_name = data.get('scenario')
    if not scenario_name:
        return jsonify({"code": 1, "message": "请指定场景名"}), 400

    # 生成唯一会话ID
    session_id = str(uuid.uuid4())
    start_msg = ds.start(scenario_name, session_id)
    SESSIONS[session_id] = {"scenario": scenario_name, "is_ended": False}

    return jsonify({
        "code": 0,
        "session_id": session_id,
        "message": start_msg
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天消息"""
    data = request.get_json()
    session_id = data.get('session_id')
    message = data.get('message', '').strip()

    if not session_id or session_id not in SESSIONS:
        return jsonify({"code": 1, "message": "会话不存在"}), 400
    if not message:
        return jsonify({"code": 1, "message": "消息不能为空"}), 400

    # 调用对话系统
    response, is_ended = ds.chat(session_id, message)
    SESSIONS[session_id]["is_ended"] = is_ended

    return jsonify({
        "code": 0,
        "message": response,
        "is_ended": is_ended
    })


# ---------------------- 前端页面 ----------------------
@app.route('/')
def index():
    """聊天界面"""
    scenarios = ds.list_scenarios()
    return render_template_string(HTML_TEMPLATE, scenarios=scenarios)


# ---------------------- 启动服务 ----------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)