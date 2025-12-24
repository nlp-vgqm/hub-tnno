# -*- coding: utf-8 -*-

import torch
import json
import argparse
from config import Config
from model import TorchModel
from transformers import BertTokenizer

"""
模型预测脚本
"""


class NERPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path

        # 加载BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

        # 加载schema映射
        self.label_to_id = self.load_schema(config["schema_path"])
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # 加载模型
        self.model = self.load_model()

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载完成，使用设备: {self.device}")

    def load_schema(self, schema_path):
        """加载标签schema"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema

    def load_model(self):
        """加载训练好的模型"""
        model = TorchModel(self.config)

        # 加载模型权重
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(self.model_path))
        else:
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        return model

    def preprocess(self, text):
        """预处理文本"""
        # 使用BERT tokenizer编码
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0).to(self.device)
        attention_mask = encoding["attention_mask"].squeeze(0).to(self.device)

        return input_ids, attention_mask, encoding

    def predict(self, text):
        """预测单个文本"""
        # 预处理
        input_ids, attention_mask, encoding = self.preprocess(text)

        # 模型预测
        with torch.no_grad():
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # 获取预测结果
            predictions = self.model(input_ids, attention_mask=attention_mask)

            # 处理CRF和非CRF的不同输出格式
            if self.config["use_crf"]:
                # CRF输出是解码后的序列
                pred_sequence = predictions[0]  # 取第一个batch
            else:
                # 非CRF输出需要argmax
                predictions = torch.argmax(predictions, dim=-1)
                pred_sequence = predictions[0].cpu().detach().tolist()

        # 解码实体
        entities = self.decode(text, encoding, pred_sequence)

        return entities, pred_sequence

    def decode(self, text, encoding, pred_sequence):
        """解码预测结果"""
        # 将token_ids转换为tokens
        token_ids = encoding["input_ids"].squeeze(0).tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # 过滤特殊标记和padding
        valid_tokens = []
        valid_labels = []

        for i, token in enumerate(tokens):
            # 跳过特殊标记和padding
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # 处理子词
            if token.startswith("##"):
                if valid_tokens:
                    valid_tokens[-1] += token[2:]
            else:
                valid_tokens.append(token)
                # 获取对应的标签
                if i < len(pred_sequence):
                    label_id = pred_sequence[i]
                    valid_labels.append(self.id_to_label.get(label_id, "O"))

        # 提取实体
        entities = self.extract_entities(valid_tokens, valid_labels)

        return entities

    def extract_entities(self, tokens, labels):
        """从token和标签中提取实体"""
        entities = []
        current_entity = None
        current_entity_tokens = []

        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                # 如果已经有实体在构建中，先保存
                if current_entity is not None:
                    entity_text = "".join(current_entity_tokens)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity,
                        "start": None,  # 这里简化处理，实际可以计算位置
                        "end": None
                    })

                # 开始新的实体
                current_entity = label[2:]  # 去掉"B-"
                current_entity_tokens = [token]

            elif label.startswith("I-"):
                # 继续当前实体
                if current_entity is not None and label[2:] == current_entity:
                    current_entity_tokens.append(token)
                else:
                    # 如果标签不匹配，结束当前实体
                    if current_entity is not None:
                        entity_text = "".join(current_entity_tokens)
                        entities.append({
                            "text": entity_text,
                            "type": current_entity,
                            "start": None,
                            "end": None
                        })
                    current_entity = None
                    current_entity_tokens = []

            else:
                # "O"标签，结束当前实体
                if current_entity is not None:
                    entity_text = "".join(current_entity_tokens)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity,
                        "start": None,
                        "end": None
                    })
                current_entity = None
                current_entity_tokens = []

        # 处理最后一个实体
        if current_entity is not None:
            entity_text = "".join(current_entity_tokens)
            entities.append({
                "text": entity_text,
                "type": current_entity,
                "start": None,
                "end": None
            })

        return entities

    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            entities, _ = self.predict(text)
            results.append({
                "text": text,
                "entities": entities
            })
        return results

    def visualize(self, text, entities):
        """可视化实体标注结果"""
        # 创建一个带颜色标注的文本
        colors = {
            "PERSON": "\033[91m",  # 红色
            "LOCATION": "\033[92m",  # 绿色
            "ORGANIZATION": "\033[93m",  # 黄色
            "TIME": "\033[94m",  # 蓝色
        }
        reset_color = "\033[0m"

        # 如果没有实体，直接返回原文本
        if not entities:
            print(f"文本: {text}")
            print("未识别到实体")
            return

        # 创建一个标记文本
        marked_text = text
        offset = 0

        for entity in entities:
            start = text.find(entity["text"])
            if start == -1:
                continue

            end = start + len(entity["text"])
            color = colors.get(entity["type"], reset_color)

            # 插入颜色标记
            marked_text = (
                    marked_text[:start + offset] +
                    f"{color}{entity['text']}{reset_color}" +
                    marked_text[end + offset:]
            )

            # 更新偏移量
            offset += len(color) + len(reset_color)

        print(f"文本: {marked_text}")
        print("\n识别到的实体:")
        for entity in entities:
            print(f"  {entity['type']}: {entity['text']}")


def main():
    parser = argparse.ArgumentParser(description="NER模型预测")
    parser.add_argument("--model_path", type=str, default="model_output/epoch_20.pth",
                        help="模型路径")
    parser.add_argument("--text", type=str, default=None,
                        help="要预测的文本")
    parser.add_argument("--file", type=str, default=None,
                        help="包含多行文本的文件路径")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式模式")

    args = parser.parse_args()

    # 加载配置
    config = Config

    # 创建预测器
    predictor = NERPredictor(config, args.model_path)

    # 交互式模式
    if args.interactive:
        print("进入交互式NER预测模式（输入'quit'退出）")
        print("=" * 50)

        while True:
            text = input("\n请输入要分析的文本: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("退出预测程序")
                break

            if not text:
                print("输入不能为空")
                continue

            try:
                entities, _ = predictor.predict(text)
                predictor.visualize(text, entities)
            except Exception as e:
                print(f"预测出错: {e}")

    # 文件模式
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            print(f"从文件 {args.file} 读取了 {len(texts)} 个文本")
            print("=" * 50)

            for i, text in enumerate(texts, 1):
                print(f"\n文本 {i}:")
                entities, _ = predictor.predict(text)
                predictor.visualize(text, entities)
                print("-" * 50)

        except FileNotFoundError:
            print(f"文件 {args.file} 不存在")

    # 单文本模式
    elif args.text:
        entities, _ = predictor.predict(args.text)
        predictor.visualize(args.text, entities)

    else:
        # 如果没有指定任何模式，运行示例
        print("使用示例文本:")
        print("=" * 50)

        example_texts = [
            "张三在北京的清华大学读书，他计划在2023年毕业。",
            "李四将于下周一在上海的阿里巴巴公司参加会议。",
            "王五在纽约时报上看到了一篇关于人工智能的文章。"
        ]

        for text in example_texts:
            print(f"\n示例文本: {text}")
            entities, _ = predictor.predict(text)
            print("识别到的实体:")
            for entity in entities:
                print(f"  {entity['type']}: {entity['text']}")
            print("-" * 50)


if __name__ == "__main__":
    main()