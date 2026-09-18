"""
train_lora.py

Pilot LoRA fine-tune of Qwen2.5-Coder-7B-Instruct on your hand-written
wiring+code dataset (pilot_dataset.jsonl). This is intentionally small
and fast -- the goal is to prove the pipeline works end to end, not to
produce a great model yet.

Usage:
    python train_lora.py

Adjust MODEL_NAME to a smaller model (e.g. "unsloth/Qwen2.5-Coder-1.5B-Instruct")
if you hit out-of-memory errors on an 8GB GPU.
"""

import json
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ----- Config -----
MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"  # pre-quantized, faster download
MAX_SEQ_LENGTH = 768
DATASET_PATH = "full_dataset.jsonl"
OUTPUT_DIR = "pilot_lora_adapter"

LORA_RANK = 16
LORA_ALPHA = 16
NUM_EPOCHS = 3          # pilot run: few epochs over a tiny dataset is fine
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4


def load_dataset(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {path}")
    return Dataset.from_list(examples)


def format_example(example, tokenizer):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    print("Loading base model (this downloads ~4-5GB the first time)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # memory saving for 8GB cards
        random_state=42,
    )

    raw_dataset = load_dataset(DATASET_PATH)
    dataset = raw_dataset.map(lambda ex: format_example(ex, tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone. Run inference_test.py next to try the fine-tuned model.")


if __name__ == "__main__":
    main()
