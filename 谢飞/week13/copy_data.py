# -*- coding: utf-8 -*-

"""
数据文件复制脚本
如果数据文件不存在，从week09目录复制
"""

import os
import shutil

def copy_data_files():
    """复制数据文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    week09_base = os.path.join(os.path.dirname(base_dir), "week09")
    
    files_to_copy = {
        "schema.json": ("data/schema.json", "data/schema.json"),
        "train": ("data/train", "data/train"),
        "test": ("data/test", "data/test"),
    }
    
    copied = False
    for name, (source_rel, target_rel) in files_to_copy.items():
        target_path = os.path.join(base_dir, target_rel)
        source_path = os.path.join(week09_base, source_rel)
        
        if not os.path.exists(target_path):
            if os.path.exists(source_path):
                try:
                    # 确保目标目录存在
                    target_dir = os.path.dirname(target_path)
                    if target_dir and not os.path.exists(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(source_path, target_path)
                    print(f"已复制文件: {name}")
                    copied = True
                except Exception as e:
                    print(f"复制文件失败 {name}: {e}")
            else:
                print(f"源文件不存在: {source_path}")
        else:
            print(f"文件已存在: {target_path}")
    
    if copied:
        print("数据文件复制完成！")
    else:
        print("所有数据文件已存在，无需复制。")

if __name__ == "__main__":
    copy_data_files()
