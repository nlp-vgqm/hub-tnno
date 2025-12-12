# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
数据文件设置脚本
将数据文件从原始位置复制到当前目录
"""

import os
import shutil
import sys

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 源文件路径
source_base = r"E:\AI\lessons\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题"
target_base = os.path.dirname(os.path.abspath(__file__))

# 需要复制的文件
files_to_copy = [
    ("data/train.json", "data/train.json"),
    ("data/valid.json", "data/valid.json"),
    ("chars.txt", "chars.txt"),
]

# 创建data目录
data_dir = os.path.join(target_base, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("创建目录: data")

# 复制文件
for source_rel, target_rel in files_to_copy:
    source_path = os.path.join(source_base, source_rel)
    target_path = os.path.join(target_base, target_rel)
    
    if os.path.exists(source_path):
        try:
            # 使用二进制模式复制大文件
            with open(source_path, 'rb') as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            print(f"[OK] 复制成功: {target_rel}")
        except Exception as e:
            print(f"[ERROR] 复制失败: {target_rel} - {e}")
    else:
        print(f"[ERROR] 源文件不存在: {source_path}")

print("\n数据文件设置完成！")

