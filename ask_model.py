"""
ask_model.py

Ask your fine-tuned 7B model any question you want, interactively --
not one of the fixed eval/inference-test questions.

Usage:
    python ask_model.py
    (then type your question when prompted, or edit QUESTION below
    to hardcode one)
"""

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
ADAPTER_DIR = "7b_lora_adapter"
MAX_SEQ_LENGTH = 2048

def main():
    print("Loading model + adapter (takes ~20-30s)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model.load_adapter(ADAPTER_DIR, adapter_name="candidate")
    FastLanguageModel.for_inference(model)
    print("Ready.\n")

    while True:
        question = input("Ask a wiring/code question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(input_ids=inputs, max_new_tokens=1300, do_sample=False)
        answer = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(answer)
        print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
