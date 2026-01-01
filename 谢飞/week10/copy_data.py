# -*- coding: utf-8 -*-
"""临时脚本：复制数据文件到本地"""
import shutil
import os

source = r"E:\AI\lessons\week10 文本生成问题\week10 文本生成问题\transformers-生成文章标题\sample_data.json"
target_dir = "data"
target_train = os.path.join(target_dir, "train.json")
target_valid = os.path.join(target_dir, "valid.json")

if os.path.exists(source):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, target_train)
    shutil.copy2(source, target_valid)
    print(f"数据文件已复制到 {target_train} 和 {target_valid}")
else:
    print(f"源文件不存在: {source}")

