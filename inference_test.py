"""
inference_test.py

Loads your fine-tuned pilot LoRA adapter and lets you ask it wiring/code
questions interactively, to manually judge whether the pilot worked.

Usage:
    python inference_test.py
"""

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
ADAPTER_DIR = "pilot_lora_adapter"
MAX_SEQ_LENGTH = 2048

# A few questions NOT in the pilot dataset -- good sanity checks
TEST_QUESTIONS = [
    "How do I wire a PIR motion sensor to an ESP32 and write code to detect motion?",
    "What's the wiring and code for an SD card module connected to an ESP32 over SPI?",
    "I want to connect a servo motor to an ESP32 and sweep it back and forth. Wiring and code?",
]


def main():
    print("Loading base model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model.load_adapter(ADAPTER_DIR, adapter_name="pilot")
    FastLanguageModel.for_inference(model)

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print("=" * 70)

        messages = [{"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=600,
            temperature=0.3,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(response)

    print("\n" + "=" * 70)
    print("Manual check: does the output follow the wiring-table + code format?")
    print("Are pin assignments at least plausible? That's a pass for the pilot stage.")


if __name__ == "__main__":
    main()
