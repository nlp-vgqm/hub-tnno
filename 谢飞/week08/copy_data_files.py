# -*- coding: utf-8 -*-
"""
复制数据文件到当前目录
"""

import os
import shutil
import sys

# 设置UTF-8编码输出
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# 源文件路径
SOURCE_BASE = r"E:\AI\lessons\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题"
TARGET_BASE = os.path.dirname(os.path.abspath(__file__))

# 需要复制的文件
files_to_copy = [
    ("data/train.json", "data/train.json"),
    ("data/valid.json", "data/valid.json"),
    ("chars.txt", "chars.txt"),
]

# 创建data目录
data_dir = os.path.join(TARGET_BASE, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"创建目录: data")

print("开始复制文件...")
print("=" * 60)

success_count = 0
fail_count = 0

for source_rel, target_rel in files_to_copy:
    source_path = os.path.join(SOURCE_BASE, source_rel)
    target_path = os.path.join(TARGET_BASE, target_rel)
    
    if os.path.exists(source_path):
        try:
            # 确保目标目录存在
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            file_size = os.path.getsize(target_path) / (1024 * 1024)  # MB
            print(f"[成功] {target_rel} ({file_size:.2f} MB)")
            success_count += 1
        except Exception as e:
            print(f"[失败] {target_rel} - {e}")
            fail_count += 1
    else:
        print(f"[错误] 源文件不存在: {source_path}")
        fail_count += 1

print("=" * 60)
print(f"复制完成: 成功 {success_count} 个, 失败 {fail_count} 个")

if success_count == len(files_to_copy):
    print("\n所有文件已成功复制！现在可以运行 python main.py 开始训练。")
else:
    print("\n部分文件复制失败，请检查错误信息。")

