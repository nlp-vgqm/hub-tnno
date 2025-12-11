# -*- coding: utf-8 -*-
"""快速复制数据文件"""

import shutil
import os

# 源和目标路径
source_dir = r"E:\AI\lessons\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题"
target_dir = os.path.dirname(os.path.abspath(__file__))

files = [
    (os.path.join(source_dir, "data", "train.json"), os.path.join(target_dir, "data", "train.json")),
    (os.path.join(source_dir, "data", "valid.json"), os.path.join(target_dir, "data", "valid.json")),
    (os.path.join(source_dir, "chars.txt"), os.path.join(target_dir, "chars.txt")),
]

print("开始复制文件...")
for src, dst in files:
    if os.path.exists(src):
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # 复制文件
            shutil.copy2(src, dst)
            print(f"成功: {os.path.basename(dst)}")
        except Exception as e:
            print(f"失败: {os.path.basename(dst)} - {e}")
    else:
        print(f"源文件不存在: {src}")

print("完成！")

