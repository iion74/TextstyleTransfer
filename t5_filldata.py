# ✅ 설치 필요시
# pip install transformers datasets

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA 사용:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥️ GPU:", torch.cuda.get_device_name(0))

# ✅ 모델 및 토크나이저 로드
model_name = "paust/pko-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# ✅ 데이터셋 로드
df = pd.read_csv(r"C:\Users\AI-LJH\Desktop\cap\smilestyle_dataset_filtered_cleaned.tsv", sep="\t")
df = df.dropna(thresh=3)

# ✅ 학습 샘플 생성 함수
def build_samples(df):
    samples = []
    for _, row in df.iterrows():
        row = row.dropna()
        for tgt_col in row.index:
            tgt = row[tgt_col]
            for src_col in row.index:
                if src_col == tgt_col:
                    continue
                src = row[src_col]
                prompt = f"Translate to {tgt_col} style: {src}"
                samples.append((prompt, tgt))
    return samples

samples = build_samples(df)

# ✅ 커스텀 Dataset
class StyleDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        inputs = self.tokenizer(
            src, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return {k: v.squeeze() for k, v in inputs.items()}

# ✅ 학습/검증 분할
train_samples, eval_samples = train_test_split(samples, test_size=0.1, random_state=42)
train_dataset = StyleDataset(train_samples, tokenizer)
eval_dataset = StyleDataset(eval_samples, tokenizer)

# ✅ 100 step마다 실시간 로그 출력 콜백
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        current = state.global_step
        total = state.max_steps
        if logs is not None:
            if "loss" in logs:
                print(f"🟠 Step {current:>5} / {total}: Train Loss = {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"🟢 Step {current:>5} / {total}: Eval  Loss = {logs['eval_loss']:.4f}")


# ✅ 학습 인자
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_style_fildata_finetuned",
    eval_strategy="steps",          
    eval_steps=1000,                      # <- 1000 스텝마다 평가
    save_strategy="steps",
    save_steps=1000,
    logging_steps=1000,
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=20,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
    disable_tqdm=False,
)


# ✅ Trainer 구성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[PrintLossCallback()]
)

# ✅ 학습 시작
trainer.train()

# ✅ 모델 저장
trainer.save_model("./t5_style_finetuned/final_model")
tokenizer.save_pretrained("./t5_style_finetuned/final_model")
