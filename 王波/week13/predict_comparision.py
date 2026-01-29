# -*- coding: utf-8 -*-

import torch
import json
import os
import argparse
import time
from config import Config
from model import TorchModel
from transformers import BertTokenizer, BertModel
from peft import PeftModel, PeftConfig

"""
模型预测脚本 - 支持LoRA和非LoRA模型对比
"""


class NERPredictor:
    def __init__(self, config, model_path=None, lora_path=None, use_lora=False, name="Model"):
        self.config = config
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_lora = use_lora
        self.name = name

        # 加载BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

        # 加载schema映射
        self.label_to_id = self.load_schema(config["schema_path"])
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # 定义标签映射（与训练一致）
        self.label_mapping = {
            0: "B-PERSON", 1: "B-LOCATION", 2: "B-ORGANIZATION", 3: "B-TIME",
            4: "I-PERSON", 5: "I-LOCATION", 6: "I-ORGANIZATION", 7: "I-TIME",
            8: "O"
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        # 加载模型
        self.model = self.load_model()

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"[{self.name}] 加载完成，使用设备: {self.device}")
        print(f"[{self.name}] 使用LoRA: {self.use_lora}")

    def load_schema(self, schema_path):
        """加载标签schema"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema

    def load_model(self):
        """加载训练好的模型（支持LoRA）"""
        print(f"[{self.name}] 开始加载模型...")

        # 如果是LoRA模型
        if self.use_lora and self.lora_path:
            print(f"[{self.name}] 加载LoRA微调模型...")
            print(f"  基础BERT模型: {self.config['bert_path']}")
            print(f"  LoRA权重路径: {self.lora_path}")

            # 方法1：直接加载完整的TorchModel（包含LoRA）
            # 创建模型实例
            model = TorchModel(self.config)

            # 加载模型权重
            if os.path.isdir(self.lora_path):
                # 如果lora_path是一个目录，检查其中是否有保存的模型
                if os.path.exists(os.path.join(self.lora_path, "adapter_model.bin")):
                    print(f"  从 {self.lora_path} 加载LoRA适配器")
                    # 使用PeftModel加载LoRA权重
                    peft_config = PeftConfig.from_pretrained(self.lora_path)
                    base_model = BertModel.from_pretrained(self.config["bert_path"])
                    peft_model = PeftModel.from_pretrained(base_model, self.lora_path)
                    model.bert = peft_model

                    # 加载分类头权重
                    if self.model_path and os.path.exists(self.model_path):
                        classifier_state = torch.load(self.model_path, map_location='cpu')
                        if 'fc' in classifier_state:
                            model.fc.load_state_dict(classifier_state['fc'])
                        if 'classify' in classifier_state:
                            model.classify.load_state_dict(classifier_state['classify'])
                        if 'crf_layer' in classifier_state and self.config["use_crf"]:
                            model.crf_layer.load_state_dict(classifier_state['crf_layer'])
                        print(f"  分类头权重从 {self.model_path} 加载")
                else:
                    print(f"  警告: {self.lora_path} 中未找到adapter_model.bin")
                    # 尝试加载整个模型状态
                    model_files = [f for f in os.listdir(self.lora_path) if f.endswith('.pth')]
                    if model_files:
                        model_path = os.path.join(self.lora_path, model_files[0])
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        print(f"  模型权重从 {model_path} 加载")
            elif self.model_path and os.path.exists(self.model_path):
                # 如果提供了完整的模型文件路径
                model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                print(f"  模型权重从 {self.model_path} 加载")

        elif self.model_path and os.path.exists(self.model_path):
            # 非LoRA模型或完整的模型文件
            print(f"[{self.name}] 加载完整模型权重: {self.model_path}")
            model = TorchModel(self.config)
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        else:
            # 创建新模型
            print(f"[{self.name}] 创建新模型（无预训练权重）")
            model = TorchModel(self.config)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{self.name}] 模型总参数: {total_params:,}")
        print(f"[{self.name}] 可训练参数: {trainable_params:,}")
        if total_params > 0:
            print(f"[{self.name}] 参数压缩比: {(trainable_params / total_params) * 100:.2f}%")

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
        # 记录推理时间
        start_time = time.time()

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

        # 计算推理时间
        inference_time = time.time() - start_time

        return entities, pred_sequence, inference_time

    def decode(self, text, encoding, pred_sequence):
        """解码预测结果，正确处理BERT分词"""
        # 将token_ids转换为tokens
        token_ids = encoding["input_ids"].squeeze(0).tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        attention_mask = encoding["attention_mask"].squeeze(0).tolist()

        # 过滤特殊标记和padding
        valid_tokens = []
        valid_labels = []

        for i, (token, mask_val) in enumerate(zip(tokens, attention_mask)):
            if mask_val == 0:  # padding
                continue
            if token in ["[CLS]", "[SEP]"]:  # 特殊标记
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
                    if label_id < 8:  # 实体标签
                        valid_labels.append(self.label_mapping.get(label_id, "O"))
                    else:
                        valid_labels.append("O")

        # 确保长度一致
        if len(valid_labels) < len(valid_tokens):
            valid_labels.extend(["O"] * (len(valid_tokens) - len(valid_labels)))

        # 提取实体
        entities = self.extract_entities(valid_tokens, valid_labels)

        return entities

    def extract_entities(self, tokens, labels):
        """从token和标签中提取实体"""
        entities = []
        current_entity = None
        current_entity_tokens = []
        current_entity_type = None

        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                # 如果已经有实体在构建中，先保存
                if current_entity is not None:
                    entity_text = "".join(current_entity_tokens)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity_type,
                        "start": i - len(current_entity_tokens),
                        "end": i - 1
                    })

                # 开始新的实体
                current_entity_type = label[2:]  # 去掉"B-"
                current_entity_tokens = [token]
                current_entity = label

            elif label.startswith("I-"):
                # 继续当前实体
                if current_entity is not None and label[2:] == current_entity_type:
                    current_entity_tokens.append(token)
                else:
                    # 如果标签不匹配，结束当前实体
                    if current_entity is not None:
                        entity_text = "".join(current_entity_tokens)
                        entities.append({
                            "text": entity_text,
                            "type": current_entity_type,
                            "start": i - len(current_entity_tokens),
                            "end": i - 1
                        })
                    current_entity = None
                    current_entity_tokens = []
                    current_entity_type = None

            else:
                # "O"标签，结束当前实体
                if current_entity is not None:
                    entity_text = "".join(current_entity_tokens)
                    entities.append({
                        "text": entity_text,
                        "type": current_entity_type,
                        "start": i - len(current_entity_tokens),
                        "end": i - 1
                    })
                current_entity = None
                current_entity_tokens = []
                current_entity_type = None

        # 处理最后一个实体
        if current_entity is not None:
            entity_text = "".join(current_entity_tokens)
            entities.append({
                "text": entity_text,
                "type": current_entity_type,
                "start": len(tokens) - len(current_entity_tokens),
                "end": len(tokens) - 1
            })

        return entities

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
            # 在实际文本中查找实体位置
            start = text.find(entity["text"])
            if start == -1:
                # 如果找不到，跳过
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


def compare_predictions(text, base_predictor, lora_predictor):
    """对比两个模型的预测结果"""
    print("\n" + "=" * 60)
    print(f"对比预测结果: {text}")
    print("=" * 60)

    # 使用基础模型预测
    base_entities, _, base_time = base_predictor.predict(text)
    print(f"\n[{base_predictor.name}] 预测结果 (推理时间: {base_time * 1000:.1f}ms):")
    if base_entities:
        for entity in base_entities:
            print(f"  {entity['type']}: {entity['text']}")
    else:
        print("  未识别到实体")

    # 使用LoRA模型预测
    lora_entities, _, lora_time = lora_predictor.predict(text)
    print(f"\n[{lora_predictor.name}] 预测结果 (推理时间: {lora_time * 1000:.1f}ms):")
    if lora_entities:
        for entity in lora_entities:
            print(f"  {entity['type']}: {entity['text']}")
    else:
        print("  未识别到实体")

    # 对比分析
    print("\n" + "-" * 60)
    print("对比分析:")

    # 收集所有实体
    all_entities = {}
    for entity in base_entities:
        key = (entity['text'], entity['type'])
        all_entities[key] = all_entities.get(key, []) + [base_predictor.name]

    for entity in lora_entities:
        key = (entity['text'], entity['type'])
        all_entities[key] = all_entities.get(key, []) + [lora_predictor.name]

    if all_entities:
        print(f"总识别实体数: {len(all_entities)}")

        # 检查一致性
        consistent_entities = []
        base_only_entities = []
        lora_only_entities = []

        for (entity_text, entity_type), models in all_entities.items():
            if len(models) == 2:
                consistent_entities.append((entity_text, entity_type))
            elif base_predictor.name in models:
                base_only_entities.append((entity_text, entity_type))
            elif lora_predictor.name in models:
                lora_only_entities.append((entity_text, entity_type))

        print(f"一致实体数: {len(consistent_entities)}")
        if consistent_entities:
            print("  一致实体: ", end="")
            for entity_text, entity_type in consistent_entities:
                print(f"{entity_text}({entity_type}) ", end="")
            print()

        print(f"仅{base_predictor.name}识别: {len(base_only_entities)}")
        if base_only_entities:
            print("  仅基础模型实体: ", end="")
            for entity_text, entity_type in base_only_entities:
                print(f"{entity_text}({entity_type}) ", end="")
            print()

        print(f"仅{lora_predictor.name}识别: {len(lora_only_entities)}")
        if lora_only_entities:
            print("  仅LoRA模型实体: ", end="")
            for entity_text, entity_type in lora_only_entities:
                print(f"{entity_text}({entity_type}) ", end="")
            print()

        # 计算一致率
        if len(all_entities) > 0:
            consistency_rate = len(consistent_entities) / len(all_entities) * 100
            print(f"\n实体一致率: {consistency_rate:.1f}%")

    # 推理时间对比
    print(f"\n推理时间对比:")
    print(f"  {base_predictor.name}: {base_time * 1000:.1f}ms")
    print(f"  {lora_predictor.name}: {lora_time * 1000:.1f}ms")

    if base_time > 0 and lora_time > 0:
        speedup = base_time / lora_time
        print(f"  速度提升: {speedup:.2f}x (LoRA相对于基础模型)")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NER模型预测对比 - LoRA vs 非LoRA")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基础模型权重路径（非LoRA）")
    parser.add_argument("--lora_model", type=str, default=None,
                        help="LoRA模型权重路径")
    parser.add_argument("--lora_weight_path", type=str, default=None,
                        help="LoRA权重路径（如果使用LoRA）")
    parser.add_argument("--text", type=str, default=None,
                        help="要预测的文本")
    parser.add_argument("--file", type=str, default=None,
                        help="包含多行文本的文件路径")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式模式")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式（同时加载两个模型进行对比）")

    args = parser.parse_args()

    # 加载配置
    config = Config.copy()

    # 如果没有提供模型路径，使用默认值
    if not args.base_model:
        args.base_model = config.get("ori_model_path", r"E:\AI_Pycharm\_code\pythonProject\MyCode_9\ner_with_Bert\model_output\epoch_20.pth")
        print(f"使用默认基础模型路径: {args.base_model}")

    if not args.lora_model and args.compare:
        # 对于对比模式，需要LoRA模型路径
        args.lora_model = os.path.join(config["model_path"], "lora_weights")
        print(f"使用默认LoRA模型路径: {args.lora_model}")

    # 检查文件是否存在
    if args.base_model and not os.path.exists(args.base_model):
        print(f"警告: 基础模型文件 {args.base_model} 不存在")
        args.base_model = None

    if args.lora_model and not os.path.exists(args.lora_model):
        print(f"警告: LoRA路径 {args.lora_model} 不存在")
        args.lora_model = None

    if args.compare:
        # 对比模式：同时加载两个模型
        print("=" * 60)
        print("NER模型对比模式 - LoRA vs 非LoRA")
        print("=" * 60)

        # 创建基础模型预测器
        print("\n[1] 加载基础模型（非LoRA）...")
        base_predictor = NERPredictor(
            config=config,
            model_path=args.base_model,
            use_lora=False,
            name="Base Model (Non-LoRA)"
        )

        # 创建LoRA模型预测器
        print("\n[2] 加载LoRA模型...")
        lora_predictor = NERPredictor(
            config=config,
            model_path=args.lora_model,
            lora_path=args.lora_weight_path if args.lora_weight_path else args.lora_model,
            use_lora=True,
            name="LoRA Model"
        )

        print("\n" + "=" * 60)
        print("模型加载完成，开始对比测试...")
        print("=" * 60)

        # 准备测试文本
        texts = []

        if args.text:
            texts.append(args.text)

        if args.file and os.path.exists(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])

        if not texts:
            # 使用示例文本
            print("使用示例文本进行对比测试...")
            texts = [
                "张三在北京的清华大学读书，他计划在2023年毕业。",
                "李四将于下周一在上海的阿里巴巴公司参加会议。",
                "王五在纽约时报上看到了一篇关于人工智能的文章。",
                "苹果公司首席执行官蒂姆·库克于昨日访问了中国北京。"
            ]

        # 对每个文本进行对比预测
        for text in texts:
            compare_predictions(text, base_predictor, lora_predictor)

        # 交互式模式
        if args.interactive or (not args.text and not args.file):
            print("进入交互式对比模式 (输入'quit'退出)")
            while True:
                user_input = input("\n请输入要分析的文本: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input:
                    compare_predictions(user_input, base_predictor, lora_predictor)
                else:
                    print("请输入有效的文本")
    else:
        # 原始模式：只加载一个模型
        print("初始化预测器...")
        predictor = NERPredictor(config, args.model_path, args.lora_path)

        # 如果没有指定任何模式，运行示例
        print("使用示例文本:")
        print("=" * 50)

        example_texts = [
            "张三在北京的清华大学读书，他计划在2023年毕业。",
            "李四将于下周一在上海的阿里巴巴公司参加会议。",
            "王五在纽约时报上看到了一篇关于人工智能的文章。",
            "苹果公司首席执行官蒂姆·库克于昨日访问了中国北京。"
        ]

        for text in example_texts:
            print(f"\n示例文本: {text}")
            entities, _, inference_time = predictor.predict(text)
            print(f"推理时间: {inference_time * 1000:.1f}ms")
            print("识别到的实体:")
            for entity in entities:
                print(f"  {entity['type']}: {entity['text']}")
            print("-" * 50)


if __name__ == "__main__":
    main()