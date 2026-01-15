
测试Seq2Seq SFT微调后的模型


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = .output_seq2seq  # 微调后的seq2seq模型路径
# 未微调时可直接使用原始模型测试
# MODEL_PATH = google-t5t5-small

def test_seq2seq_model()
    print(正在加载Seq2Seq模型...)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=auto if torch.cuda.is_available() else None
    )
    
    # 测试问题
    test_questions = [
        介绍一下人工智能,
        什么是深度学习,
        Python列表和元组的区别
    ]
    
    print(n开始测试Seq2Seq模型...)
    print(=  50)
    
    for question in test_questions
        # 构造输入（与训练时格式一致，带question 前缀）
        input_text = fquestion {question}
        inputs = tokenizer(
            input_text,
            max_length=128,
            padding=max_length,
            truncation=True,
            return_tensors=pt
        )
        
        # 移动到GPU（如果可用）
        if torch.cuda.is_available()
            inputs = {k v.cuda() for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad()
            outputs = model.generate(
                inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码结果
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f问题 {question})
        print(f回答 {response})
        print(-  50)

if __name__ == __main__
    test_seq2seq_model()