import json
import os

def load_json_file(file_path):
    """
    读取并验证JSON文件，返回解析后的字典/列表，若失败则输出详细错误
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}，请检查文件路径！")
        return None
    
    # 2. 检查文件是否为空
    if os.path.getsize(file_path) == 0:
        print(f"错误：{file_path} 文件为空，不是有效的JSON！")
        return None
    
    try:
        # 3. 读取文件（指定UTF-8编码，兼容BOM头）
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # 4. 解析JSON并捕获具体错误
            try:
                data = json.load(f)
                print(f"成功：{file_path} JSON格式验证通过！")
                return data
            except json.JSONDecodeError as e:
                # 输出具体的解析错误（位置+原因）
                print(f"错误：JSON格式无效！具体原因：{e.msg}")
                print(f"错误位置：行 {e.lineno}，列 {e.colno}")
                return None
    except Exception as e:
        # 捕获其他异常（如编码错误、权限问题）
        print(f"读取文件时出错：{str(e)}")
        return None

# ==================== 调用示例 ====================
# 替换成你的data.json实际路径（建议用绝对路径，避免路径错误）
json_path = "data.json"  # 若文件在其他文件夹，写绝对路径如：C:/project/data.json
json_data = load_json_file(json_path)

# 如果解析成功，可继续后续操作
if json_data is not None:
    print("JSON内容：", json_data)
