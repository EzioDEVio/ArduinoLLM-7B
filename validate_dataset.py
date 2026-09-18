"""
validate_dataset.py

Runs BEFORE you merge new generated examples into your training set.
Checks every example for:
  - Format compliance (wiring table, ASCII diagram, code block, explanation)
  - Runtime contamination (wrong-framework API leaking in, e.g. RPi.GPIO in
    an Arduino answer) -- only possible for examples tagged with board/language
  - Basic sanity (non-trivial length, has a real code block)

Splits input into two output files:
  - <input>_validated.jsonl  -- examples that passed everything, safe to merge
  - <input>_rejected.jsonl   -- examples that failed, with reasons, for review

Examples from before the board/language tagging was added (your original
pilot_dataset.jsonl and the first 82-task generation batch) won't have
"board"/"language" keys -- these still get format-checked, but contamination
checking is skipped for them with a note, rather than failing them outright.

Usage:
    python validate_dataset.py generated_dataset.jsonl
"""

import json
import sys
import re

REQUIRED_SECTIONS = ["Wiring Table", "Wiring Diagram", "```", "How it works"]
MIN_OUTPUT_LENGTH = 200  # characters -- catches suspiciously short/truncated examples

RUNTIME_SIGNATURES = {
    "cpp": {
        "positive": ["void setup()", "void loop()", "pinMode", "digitalWrite"],
        "negative": ["machine.Pin", "machine.I2C", "RPi.GPIO", "import smbus",
                     "def setup", "GPIO.setmode", "physical pin", "import board",
                     "import digitalio"],
    },
    "python": {
        # Expanded: real Raspberry Pi Python code legitimately uses many
        # different libraries depending on the bus/peripheral (SPI->spidev,
        # camera->picamera2, LEDs->rpi_ws281x, serial->pyserial, 1-wire->
        # kernel filesystem with no special import at all). A narrow list
        # here caused false-positive rejections -- this check is now
        # informational only (see "positive_is_advisory" below), so a
        # generous list is fine.
        "positive": ["import RPi.GPIO", "GPIO.setmode", "smbus", "gpiozero",
                     "spidev", "SpiDev", "picamera2", "rpi_ws281x", "PixelStrip",
                     "import serial", "Serial(", "/sys/bus/w1", "w1_slave",
                     "sounddevice", "pyaudio"],
        "negative": ["machine.Pin", "machine.I2C", "from machine import",
                     "void setup()", "void loop()", "#include", "import board",
                     "import digitalio"],
        "positive_is_advisory": True,
    },
    "micropython": {
        "positive": ["machine.Pin", "machine.I2C", "from machine import",
                     "machine.ADC", "machine.PWM"],
        "negative": ["RPi.GPIO", "import smbus", "GPIO.setmode", "gpiozero",
                     "void setup()", "void loop()", "#include", "import board",
                     "import digitalio"],
    },
    "circuitpython": {
        "positive": ["import board", "import digitalio", "import busio", "board."],
        "negative": ["RPi.GPIO", "import smbus", "GPIO.setmode", "gpiozero",
                     "void setup()", "void loop()", "#include",
                     "from machine import", "machine.Pin"],
    },
}


def extract_actual_code_block(text):
    match = re.search(r"\*\*Code\*\*\s*```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


def validate_example(example, index):
    issues = []
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    board = example.get("board")
    language = example.get("language")

    if not instruction or not output:
        issues.append("missing instruction or output field")
        return issues  # nothing else worth checking

    if len(output) < MIN_OUTPUT_LENGTH:
        issues.append(f"output suspiciously short ({len(output)} chars, expected {MIN_OUTPUT_LENGTH}+)")

    for section in REQUIRED_SECTIONS:
        if section not in output:
            issues.append(f"missing required section: {section}")

    code = extract_actual_code_block(output)
    if not code or not code.strip():
        issues.append("no code block found after '**Code**' heading")

    if language and language in RUNTIME_SIGNATURES:
        sig = RUNTIME_SIGNATURES[language]
        for bad in sig["negative"]:
            if bad in output:
                issues.append(f"runtime contamination: found '{bad}' (wrong for language={language})")
        if not any(good in output for good in sig["positive"]):
            if sig.get("positive_is_advisory"):
                issues.append(f"NOTE: no recognized {language} library signature found "
                              f"(informational only -- code may still be correct)")
            else:
                issues.append(f"missing expected runtime signature for language={language} "
                              f"(none of {sig['positive']} found)")
    else:
        issues.append("NOTE: no board/language tag -- contamination check skipped "
                      "(this is informational, not a failure)")

    return issues


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <dataset.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    base = input_path.rsplit(".", 1)[0]
    validated_path = f"{base}_validated.jsonl"
    rejected_path = f"{base}_rejected.jsonl"

    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {input_path}\n")

    validated = []
    rejected = []

    for i, ex in enumerate(examples, 1):
        issues = validate_example(ex, i)
        # Split real problems from informational notes
        real_issues = [x for x in issues if not x.startswith("NOTE:")]
        notes = [x for x in issues if x.startswith("NOTE:")]

        if real_issues:
            rejected.append({"index": i, "instruction": ex.get("instruction", "")[:80],
                             "issues": real_issues, "example": ex})
            print(f"[{i}/{len(examples)}] REJECTED: {ex.get('instruction', '')[:60]}")
            for issue in real_issues:
                print(f"    - {issue}")
        else:
            validated.append(ex)
            status = "OK" if not notes else "OK (unchecked contamination)"
            print(f"[{i}/{len(examples)}] {status}")

    with open(validated_path, "w", encoding="utf-8") as f:
        for ex in validated:
            f.write(json.dumps(ex) + "\n")

    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total examples:  {len(examples)}")
    print(f"Validated (OK):  {len(validated)}  -> {validated_path}")
    print(f"Rejected:        {len(rejected)}  -> {rejected_path}")
    if rejected:
        print(f"\nReview {rejected_path} before deciding whether to fix or discard those examples.")
    print(f"\nOnly merge {validated_path} into your training set, not the raw generated file.")


if __name__ == "__main__":
    main()
