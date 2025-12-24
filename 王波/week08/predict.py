# -*- coding: utf-8 -*-
import torch
import os
import logging
import jieba
from loader import load_data
from config import Config
from model import SiameseNetwork

"""
模型效果测试
"""


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model

        # 设置logger
        self.logger = logging.getLogger(__name__)

        # 将模型移到合适的设备
        if torch.cuda.is_available():
            self.model = model.cuda()
            self.logger.info("使用GPU进行推理")
        else:
            self.model = model.cpu()
            self.logger.info("使用CPU进行推理")

        self.model.eval()

        # 直接使用传入的知识库数据
        self.knwb_data = knwb_data
        self.build_knowledge_base()

    def build_knowledge_base(self):
        """构建知识库向量"""
        self.logger.info("构建知识库向量...")

        # 获取数据集的词汇表和schema
        self.vocab = self.knwb_data.dataset.vocab
        self.schema = self.knwb_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())

        # 收集所有问题
        all_questions = []
        self.question_to_label = {}

        # 检查知识库数据结构
        if not hasattr(self.knwb_data.dataset, 'knwb'):
            self.logger.error("知识库数据格式不正确，缺少'knwb'属性")
            raise ValueError("知识库数据格式不正确")

        for label_idx, question_ids in self.knwb_data.dataset.knwb.items():
            for question_id in question_ids:
                all_questions.append(question_id)
                idx = len(all_questions) - 1
                self.question_to_label[idx] = label_idx

        if not all_questions:
            self.logger.error("知识库为空，无法构建预测器")
            raise ValueError("知识库为空")

        # 批量编码所有问题
        all_vectors = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(all_questions), batch_size):
                batch = all_questions[i:i + batch_size]
                batch_tensor = torch.stack(batch, dim=0)

                if torch.cuda.is_available():
                    batch_tensor = batch_tensor.cuda()

                vectors = self.model(batch_tensor, mode="encode")
                vectors = torch.nn.functional.normalize(vectors, dim=-1)

                all_vectors.append(vectors.cpu())

        self.knwb_vectors = torch.cat(all_vectors, dim=0)
        self.logger.info(f"知识库构建完成，共 {len(all_questions)} 个问题，{len(self.knwb_data.dataset.knwb)} 个类别")

    def encode_sentence(self, text):
        """编码句子为ID序列"""
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))

        # 填充或截断
        max_length = self.config["max_length"]
        input_id = input_id[:max_length]
        input_id += [0] * (max_length - len(input_id))
        return input_id

    def predict(self, sentence):
        """预测句子类别"""
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])

        if torch.cuda.is_available():
            input_id = input_id.cuda()

        with torch.no_grad():
            # 获取查询句子的向量
            query_vector = self.model(input_id, mode="encode")
            query_vector = torch.nn.functional.normalize(query_vector, dim=-1)

            # 计算与知识库中所有向量的余弦相似度
            similarities = torch.mm(query_vector, self.knwb_vectors.T)

            # 找到最相似的问题
            hit_index = int(torch.argmax(similarities.squeeze()))
            confidence = similarities.squeeze().max().item()

            # 获取对应的标签
            label_idx = self.question_to_label.get(hit_index, 0)
            label_name = self.index_to_standard_question.get(label_idx, "未知")

            # 找到最相似的几个结果（可选）
            top_k = 3
            top_values, top_indices = torch.topk(similarities.squeeze(), min(top_k, len(self.knwb_vectors)))

            top_results = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                score = top_values[i].item()
                lbl_idx = self.question_to_label.get(idx, 0)
                lbl_name = self.index_to_standard_question.get(lbl_idx, "未知")
                top_results.append((lbl_name, score))

        return label_name, confidence, top_results


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("开始初始化预测器...")

    # 尝试加载原始训练数据作为知识库
    original_data_path = Config.get("original_train_data_path")
    if not original_data_path or not os.path.exists(original_data_path):
        logger.warning(f"原始训练数据不存在: {original_data_path}")
        logger.info("尝试使用三元组数据作为知识库...")
        original_data_path = Config["train_data_path"]

    # 加载知识库数据
    logger.info(f"加载知识库数据: {original_data_path}")
    try:
        knwb_data = load_data(original_data_path, Config, shuffle=False)
        logger.info(f"知识库数据加载完成，共 {len(knwb_data.dataset)} 个样本")
    except Exception as e:
        logger.error(f"加载知识库数据失败: {e}")
        raise

    # 加载模型
    model = SiameseNetwork(Config)
    logger.info(f"模型初始化完成，参数数量: {sum(p.numel() for p in model.parameters())}")

    # 加载训练好的模型权重
    model_path = Config["model_path"]
    model_files = []

    # 首先尝试加载最佳模型
    best_model_path = os.path.join(model_path, "best_model.pth")
    if os.path.exists(best_model_path):
        model_files.append(("最佳模型", best_model_path))

    # 查找所有模型文件
    if os.path.exists(model_path):
        for f in os.listdir(model_path):
            if f.endswith(".pth"):
                full_path = os.path.join(model_path, f)
                model_files.append((f, full_path))

    # 按修改时间排序，选择最新的
    if model_files:
        # 按文件修改时间排序
        model_files.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        selected_model_name, selected_model_path = model_files[0]

        logger.info(f"加载模型: {selected_model_name}")
        try:
            model.load_state_dict(torch.load(selected_model_path, map_location='cpu'))
            logger.info(f"模型加载成功: {selected_model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.warning("使用随机初始化的模型")
    else:
        logger.warning("未找到训练好的模型，使用随机初始化的模型")

    # 创建预测器
    try:
        pd = Predictor(Config, model, knwb_data)
        logger.info("预测器初始化完成")
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        raise

    # 交互式预测
    logger.info("开始交互式预测，输入'quit'退出")

    while True:
        try:
            sentence = input("\n请输入问题（输入'quit'退出）：")
            if sentence.lower() == 'quit':
                logger.info("退出预测程序")
                break

            if not sentence.strip():
                print("请输入有效的问题")
                continue

            label, confidence, top_results = pd.predict(sentence)
            print(f"\n预测结果：{label}")
            print(f"置信度：{confidence:.4f}")

            if len(top_results) > 1:
                print("\nTop 3 结果：")
                for i, (lbl, score) in enumerate(top_results):
                    print(f"  {i + 1}. {lbl} ({score:.4f})")

            print("-" * 50)

        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            break
        except Exception as e:
            print(f"预测出错：{e}")
            import traceback

            traceback.print_exc()