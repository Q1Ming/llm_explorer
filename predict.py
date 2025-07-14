# predict.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer
from typing import Tuple

# --- 1. 配置常量 ---
# 使用常量来管理路径，更清晰，易于修改
SAVED_MODEL_PATH = "sentiment_analyzer"

#加载模型
def load_trained_model(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    从本地文件夹加载微调好的模型和分词器。
    增加了路径存在性检查，使函数更健壮。
    """
    if not os.path.isdir(model_path):
        # 提供更明确的错误信息，指导用户先运行训练脚本
        raise FileNotFoundError(
            f"模型目录 '{model_path}' 未找到。 "
            f"请先运行 'main.py' 来训练并保存模型。"
        )

    print(f"正在从 '{model_path}' 加载模型和分词器...")

    # 注意：我们现在是从一个本地目录加载，而不是从 Hugging Face Hub 下载
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("加载完成！")
    return model, tokenizer

#预测情绪
def predict_sentiment(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    使用加载好的模型对单个文本进行情感预测。
    将模型移动到设备的操作在加载后执行一次即可，无需在每次预测时都执行。
    """
    # 确定设备
    device = model.device

    # 对输入文本进行分词，并移动到正确的设备
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # 进行预测，不计算梯度以节省资源
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    # 从模型配置中获取标签名称，这比硬编码更可靠
    label = model.config.id2label.get(predicted_class_id, "UNKNOWN")

    print(f"文本: '{text}'")
    if label == "POSITIVE":
        print(">> 预测情感: 正面 👍")
    elif label == "NEGATIVE":
        print(">> 预测情感: 负面 👎")
    else:
        print(f">> 预测结果: {label}")
    print("-" * 30)

#运行交互式会话
def run_interactive_session(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """启动一个交互式会话，让用户可以连续输入并获得预测。"""
    # 将模型移动到设备的操作在这里执行一次
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"模型已移动到设备: {device.upper()}")

    print("\n情感分析预测已就绪。请输入一句话进行分析（输入 'exit' 或 Ctrl+C 退出）：")

    try:
        while True:
            user_input = input("请输入: ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue

            predict_sentiment(user_input, model, tokenizer)
    except (KeyboardInterrupt, EOFError):
        print("\n再见！")


if __name__ == "__main__":
    try:
        # 只需加载一次模型
        trained_model, trained_tokenizer = load_trained_model(SAVED_MODEL_PATH)
        # 启动交互式会话
        run_interactive_session(trained_model, trained_tokenizer)
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生了一个意外错误: {e}")
