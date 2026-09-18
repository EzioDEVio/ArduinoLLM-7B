"""
generate_dataset.py

Calls the Anthropic API to generate wiring+code training examples in the
same format your pilot dataset uses (wiring table + ASCII diagram + code
+ "how it works"), for every board/component combination in task_list.json.

Grounds each generation with the relevant board pinout data from
pinouts.json, so pin conventions stay consistent across the whole batch --
directly addressing the fragility we saw with the tiny pilot dataset.

Resumable: if interrupted, re-running skips tasks already completed
(matched by instruction text already present in the output file).

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here

Usage:
    python generate_dataset.py
"""

import json
import os
import re
import time
import sys

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic")
    sys.exit(1)

TASK_LIST_PATH = "task_list.json"
PINOUTS_PATH = "pinouts.json"
OUTPUT_PATH = "generated_dataset.jsonl"
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.0  # be polite to the API

REQUIRED_SECTIONS = ["Wiring Table", "Wiring Diagram", "```", "How it works"]

FORMAT_TEMPLATE = """You are generating ONE training example for fine-tuning a code assistant \
that specializes in embedded systems wiring and code generation. Follow this EXACT format, \
matching the style below precisely (this exact structure is what the model is being trained \
to reproduce):

---
[One sentence of context about the wiring approach -- e.g. bus choice, voltage concerns, or \
why pins are being grouped a certain way.]

**Wiring Table**

| Component Pin | Connects to {board_name} |
|---|---|
| ... | ... |

**Wiring Diagram**
```
[ASCII box-and-line diagram, in the style below, showing the board and \
component(s) as boxes with pin connections drawn as lines/labels between them]
```

[Optional: one sentence noting any voltage/safety consideration, if relevant.]

**Code**
```{language}
[complete, working code implementing this wiring]
```

**How it works:** [1-2 sentences explaining the one or two most important \
or non-obvious lines/decisions in the code -- not a restatement of what every line does.]
---

Reference pinout/board data to ground your pin choices (use these conventions \
exactly where they overlap with this task; invent reasonable additional pins \
consistent with this data where the specific component isn't listed):

{pinout_context}

CRITICAL runtime environment for this task -- get this exactly right, it is the most
important constraint: {runtime_note}

Now generate ONE example for this exact task:
Board: {board_name}
Component(s): {components}
Protocol: {protocol}
Code language: {language}

Respond with ONLY a JSON object with exactly two keys: "instruction" (a natural, \
varied-phrasing user question asking for wiring+code for this exact task) and \
"output" (the full formatted answer following the template above exactly). \
Do not include any text outside the JSON object. Do not wrap the JSON in markdown \
code fences.
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


RUNTIME_NOTES = {
    "cpp": "This is Arduino/ESP-IDF C++ code using the Arduino framework "
           "(setup()/loop(), pinMode(), digitalWrite(), Wire.begin(), etc.). "
           "Never use Python syntax or MicroPython/CircuitPython APIs.",
    "python": "This is standard Linux CPython running on a Raspberry Pi OS "
              "installation, using RPi.GPIO or gpiozero for GPIO, and smbus/smbus2 "
              "for I2C. This code assumes a full Linux OS underneath. "
              "NEVER use `machine.Pin`, `machine.I2C`, or any MicroPython/CircuitPython "
              "import -- those do not exist in this environment.",
    "micropython": "This is MicroPython running directly on a bare microcontroller "
                   "with NO operating system underneath -- use the `machine` module "
                   "(machine.Pin, machine.I2C, machine.SPI, machine.ADC, machine.PWM) "
                   "and `utime`/`time` for delays. NEVER use RPi.GPIO, smbus, or any "
                   "Linux-only Python library -- those do not exist in this environment. "
                   "Do not assume a filesystem beyond MicroPython's own limited flash storage.",
    "circuitpython": "This is Adafruit CircuitPython running directly on a "
                     "microcontroller with NO operating system underneath -- use "
                     "`board` and `digitalio`/`busio` modules (board.SCL, digitalio.DigitalInOut, "
                     "busio.I2C, etc.) and Adafruit's CircuitPython driver library conventions. "
                     "NEVER use RPi.GPIO, smbus, or the `machine` module (that's MicroPython, "
                     "a different but related runtime) -- those do not exist in this environment.",
}


def build_pinout_context(pinouts, board_name):
    context = {}
    if board_name in pinouts:
        context[board_name] = pinouts[board_name]
    if "common_modules" in pinouts:
        context["common_modules"] = pinouts["common_modules"]
    return json.dumps(context, indent=2)


def already_done_instructions(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    done.add(entry["instruction"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def validate_output(output_text):
    return all(section in output_text for section in REQUIRED_SECTIONS)


def extract_json(raw_text):
    # Model sometimes wraps in code fences despite instructions -- strip if present
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    return json.loads(raw_text)


def generate_one(client, pinouts, task):
    board_name = task["board"]
    components = ", ".join(task["components"])
    protocol = task["protocol"]
    language = task["language"]
    pinout_context = build_pinout_context(pinouts, board_name)

    runtime_note = RUNTIME_NOTES.get(language, "")
    # Markdown code fences only understand real language names -- micropython/
    # circuitpython aren't fence languages, they're our own task tags.
    fence_language = "python" if language in ("python", "micropython", "circuitpython") else language

    prompt = FORMAT_TEMPLATE.format(
        board_name=board_name.replace("_", " "),
        components=components,
        protocol=protocol,
        language=fence_language,
        pinout_context=pinout_context,
        runtime_note=runtime_note,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            parsed = extract_json(raw_text)

            if "instruction" not in parsed or "output" not in parsed:
                raise ValueError("Missing required keys in response")

            if not validate_output(parsed["output"]):
                raise ValueError(
                    f"Output missing required sections: {REQUIRED_SECTIONS}"
                )

            # Tag with board/language ourselves -- we already know these from
            # the task, no need to trust the model to echo them back correctly.
            # This is what lets validate_dataset.py check runtime contamination
            # against the CORRECT expected runtime per example.
            parsed["board"] = board_name
            parsed["language"] = language

            return parsed

        except (json.JSONDecodeError, ValueError, anthropic.APIError) as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first:")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    tasks = load_json(TASK_LIST_PATH)
    pinouts = load_json(PINOUTS_PATH)

    done_instructions_placeholder = already_done_instructions(OUTPUT_PATH)
    print(f"Found {len(done_instructions_placeholder)} already-completed examples in {OUTPUT_PATH}")
    print(f"Total tasks: {len(tasks)}")

    completed = 0
    failed = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, task in enumerate(tasks, 1):
            task_desc = f"{task['board']} + {', '.join(task['components'])}"
            print(f"\n[{i}/{len(tasks)}] {task_desc}")

            result = generate_one(client, pinouts, task)

            if result is None:
                print(f"  FAILED after {MAX_RETRIES} attempts, skipping.")
                failed += 1
                continue

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            completed += 1
            print(f"  OK -- saved.")

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
