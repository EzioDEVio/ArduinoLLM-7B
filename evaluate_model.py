"""
evaluate_model.py

Runs a held-out test set (questions never seen during training) through
the fine-tuned model and scores it automatically on:
  - Format compliance (wiring table, ASCII diagram, code block, explanation)
  - Runtime contamination (does the code use the WRONG framework's API --
    e.g. RPi.GPIO in an Arduino answer, or machine.Pin in a Raspberry Pi answer)
  - Code syntax validity (does the Python code at least parse; does the C++
    code have balanced braces and the expected Arduino skeleton)

Also measures efficiency:
  - Model load time
  - Generation speed (tokens/second)
  - Peak GPU memory used during generation
  - Adapter file size on disk

This does NOT verify that pin numbers or logic are factually correct --
that still needs human review. Treat this as an automated first pass that
catches structural and contamination failures cheaply, before you spend
time manually reading every answer.

Usage:
    python evaluate_model.py
"""

import json
import os
import time
import ast
import re

import torch
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
ADAPTER_DIR = "7b_lora_adapter"
EVAL_DATASET_PATH = "eval_dataset.jsonl"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 900

REQUIRED_SECTIONS = ["Wiring Table", "Wiring Diagram", "```", "How it works"]

# Positive signal: at least one of these should appear for the given language.
# Negative signal: NONE of these should appear -- they belong to a different runtime.
RUNTIME_SIGNATURES = {
    "cpp": {
        "positive": ["void setup()", "void loop()", "pinMode", "digitalWrite"],
        "negative": ["machine.Pin", "machine.I2C", "RPi.GPIO", "import smbus",
                     "def setup", "GPIO.setmode", "physical pin"],
    },
    "python": {  # Raspberry Pi Linux CPython
        "positive": ["import RPi.GPIO", "GPIO.setmode", "smbus", "gpiozero"],
        "negative": ["machine.Pin", "machine.I2C", "from machine import",
                     "void setup()", "void loop()", "#include"],
    },
    "micropython": {
        "positive": ["machine.Pin", "machine.I2C", "from machine import", "machine.ADC", "machine.PWM"],
        "negative": ["RPi.GPIO", "import smbus", "GPIO.setmode", "gpiozero",
                     "void setup()", "void loop()", "#include"],
    },
}


def load_eval_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def extract_code_blocks(text):
    return re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)


def extract_actual_code_block(text):
    """The format has TWO fenced blocks: the ASCII wiring diagram, then the
    real code, introduced by a '**Code**' heading. Grab the one after that
    heading specifically, rather than just the first fenced block found."""
    code_section_match = re.search(r"\*\*Code\*\*\s*```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if code_section_match:
        return code_section_match.group(1)
    # Fallback: if no "**Code**" heading found, take the LAST fenced block
    # (diagram is usually first, code usually last, when headings are missing)
    blocks = extract_code_blocks(text)
    return blocks[-1] if blocks else None


def check_python_syntax(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_cpp_sanity(code):
    # Not a real compiler check -- just catches obviously broken structure
    issues = []
    if code.count("{") != code.count("}"):
        issues.append("unbalanced braces")
    if code.count("(") != code.count(")"):
        issues.append("unbalanced parens")
    if "void setup()" not in code:
        issues.append("missing void setup()")
    if "void loop()" not in code:
        issues.append("missing void loop()")
    return (len(issues) == 0), ", ".join(issues) if issues else None


def score_answer(answer_text, language):
    result = {
        "format_score": 0,
        "format_max": len(REQUIRED_SECTIONS),
        "format_missing": [],
        "runtime_clean": True,
        "runtime_violations": [],
        "runtime_positive_found": False,
        "syntax_ok": None,
        "syntax_error": None,
    }

    for section in REQUIRED_SECTIONS:
        if section in answer_text:
            result["format_score"] += 1
        else:
            result["format_missing"].append(section)

    sig = RUNTIME_SIGNATURES.get(language, {})
    for bad in sig.get("negative", []):
        if bad in answer_text:
            result["runtime_clean"] = False
            result["runtime_violations"].append(bad)
    for good in sig.get("positive", []):
        if good in answer_text:
            result["runtime_positive_found"] = True
            break

    code_blocks = extract_code_blocks(answer_text)
    actual_code = extract_actual_code_block(answer_text)
    if actual_code:
        if language in ("python", "micropython"):
            ok, err = check_python_syntax(actual_code)
        else:
            ok, err = check_cpp_sanity(actual_code)
        result["syntax_ok"] = ok
        result["syntax_error"] = err

    return result


def main():
    questions = load_eval_questions(EVAL_DATASET_PATH)
    print(f"Loaded {len(questions)} held-out evaluation questions\n")

    print("Loading model + adapter...")
    load_start = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model.load_adapter(ADAPTER_DIR, adapter_name="candidate")
    FastLanguageModel.for_inference(model)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s\n")

    adapter_size_bytes = sum(
        os.path.getsize(os.path.join(ADAPTER_DIR, f))
        for f in os.listdir(ADAPTER_DIR)
        if os.path.isfile(os.path.join(ADAPTER_DIR, f))
    )

    results = []
    total_tokens = 0
    total_gen_time = 0.0

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['instruction'][:70]}...")

        messages = [{"role": "user", "content": q["instruction"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        torch.cuda.reset_peak_memory_stats()
        gen_start = time.time()
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        gen_time = time.time() - gen_start
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        new_tokens = outputs.shape[1] - inputs.shape[1]
        total_tokens += new_tokens
        total_gen_time += gen_time

        answer_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        score = score_answer(answer_text, q["language"])
        score["question"] = q["instruction"]
        score["board"] = q["board"]
        score["language"] = q["language"]
        score["gen_time_sec"] = round(gen_time, 2)
        score["tokens_generated"] = new_tokens
        score["peak_vram_gb"] = round(peak_mem_gb, 2)
        results.append(score)

        status = "OK" if (score["format_score"] == score["format_max"]
                           and score["runtime_clean"]) else "ISSUES"
        print(f"  [{status}] format {score['format_score']}/{score['format_max']}, "
              f"runtime_clean={score['runtime_clean']}, "
              f"syntax_ok={score['syntax_ok']}, "
              f"{new_tokens} tokens in {gen_time:.1f}s")

    # --- Summary ---
    n = len(results)
    perfect_format = sum(1 for r in results if r["format_score"] == r["format_max"])
    runtime_clean_count = sum(1 for r in results if r["runtime_clean"])
    positive_found_count = sum(1 for r in results if r["runtime_positive_found"])
    syntax_ok_count = sum(1 for r in results if r["syntax_ok"] is True)
    syntax_checked_count = sum(1 for r in results if r["syntax_ok"] is not None)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated: {n}")
    print(f"\n-- Accuracy (automated checks only -- pin correctness needs human review) --")
    print(f"Full format compliance:      {perfect_format}/{n} ({100*perfect_format/n:.0f}%)")
    print(f"No runtime contamination:    {runtime_clean_count}/{n} ({100*runtime_clean_count/n:.0f}%)")
    print(f"Correct runtime API present: {positive_found_count}/{n} ({100*positive_found_count/n:.0f}%)")
    if syntax_checked_count:
        print(f"Code syntax valid:           {syntax_ok_count}/{syntax_checked_count} "
              f"({100*syntax_ok_count/syntax_checked_count:.0f}%)")

    print(f"\n-- Efficiency --")
    print(f"Model load time:      {load_time:.1f}s")
    print(f"Avg generation speed: {total_tokens/total_gen_time:.1f} tokens/sec")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Peak VRAM during generation: {max(r['peak_vram_gb'] for r in results):.2f} GB")
    print(f"LoRA adapter size on disk: {adapter_size_bytes / (1024**2):.1f} MB")

    contaminated = [r for r in results if not r["runtime_clean"]]
    if contaminated:
        print(f"\n-- Contamination details ({len(contaminated)} question(s)) --")
        for r in contaminated:
            print(f"  [{r['board']}/{r['language']}] {r['question'][:60]}...")
            print(f"    Found wrong-runtime terms: {r['runtime_violations']}")

    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-question detail written to eval_report.json")


if __name__ == "__main__":
    main()
