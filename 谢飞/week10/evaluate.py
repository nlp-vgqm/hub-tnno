# -*- coding: utf-8 -*-

"""
模型评估和文本生成
"""

import torch
import logging
from transformers import BertTokenizer


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config, model, logger, valid_data):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = valid_data
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    
    def decode_text(self, token_ids):
        """将token ids解码为文本"""
        # 转换为列表（如果是tensor）
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        # 过滤掉特殊token（[CLS], [SEP], [PAD]）
        filtered_ids = []
        for token_id in token_ids:
            if token_id == self.tokenizer.cls_token_id:
                continue  # 跳过[CLS]
            elif token_id == self.tokenizer.sep_token_id:
                break  # 遇到[SEP]停止
            elif token_id == self.tokenizer.pad_token_id:
                break  # 遇到[PAD]停止
            else:
                filtered_ids.append(token_id)
        
        # 解码为文本
        text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
        return text
    
    def eval(self, epoch):
        """评估模型"""
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        
        with torch.no_grad():
            # 测试几个样本
            sample_count = 0
            max_samples = 5  # 每次评估显示5个样本
            
            for index, batch_data in enumerate(self.valid_data):
                if sample_count >= max_samples:
                    break
                
                # 移动到GPU（如果可用）
                if torch.cuda.is_available():
                    batch_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch_data.items()}
                
                encoder_input_ids = batch_data["encoder_input_ids"]
                encoder_attention_mask = batch_data["encoder_attention_mask"]
                labels = batch_data["labels"]
                
                # 生成文本
                generated_ids = self.model.generate(
                    encoder_input_ids=encoder_input_ids,
                    encoder_attention_mask=encoder_attention_mask,
                    max_length=self.config["output_max_length"],
                    beam_size=1  # 使用贪心解码
                )
                
                # 显示结果
                batch_size = encoder_input_ids.size(0)
                for i in range(min(batch_size, max_samples - sample_count)):
                    # 输入（content）
                    input_text = self.decode_text(encoder_input_ids[i])
                    # 生成结果
                    generated_text = self.decode_text(generated_ids[i])
                    # 真实标签（title）- 需要过滤-100（忽略的标签）
                    label_ids = labels[i].clone()
                    label_ids[label_ids == -100] = self.tokenizer.pad_token_id  # 将-100替换为PAD以便解码
                    label_text = self.decode_text(label_ids)
                    
                    self.logger.info("-" * 60)
                    self.logger.info("样本 %d:" % (sample_count + 1))
                    self.logger.info("输入（Content）: %s" % input_text[:100] + "..." if len(input_text) > 100 else input_text)
                    self.logger.info("生成（Title）: %s" % generated_text)
                    self.logger.info("真实（Title）: %s" % label_text)
                    
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
        
        self.model.train()
        return


if __name__ == "__main__":
    # 测试代码
    from config import Config
    from model import BertEncoderDecoder
    from loader import load_data
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 加载模型和数据
    model = BertEncoderDecoder(Config)
    valid_data = load_data(Config["valid_data_path"], Config, shuffle=False)
    
    evaluator = Evaluator(Config, model, logger, valid_data)
    evaluator.eval(1)

