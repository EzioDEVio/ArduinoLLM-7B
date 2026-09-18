# Arduino/Embedded Wiring & Code Assistant

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/EzioDEVio/arduino-embedded-qwen2.5-coder-7b)
[![Dataset size](https://img.shields.io/badge/dataset-373%20examples-green)](full_dataset.jsonl)
[![Base model](https://img.shields.io/badge/base%20model-Qwen2.5--Coder--7B-orange)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

A LoRA fine-tune of **Qwen2.5-Coder-7B-Instruct**, specialized for generating wiring instructions and working code for embedded/maker projects — Arduino C++, Raspberry Pi Python, MicroPython, and CircuitPython, across 7 boards.

Given a plain-language request like *"wire a BME280 to an ESP32 over I2C"*, it produces a wiring table, an ASCII wiring diagram, complete working code, and a short explanation of the non-obvious parts — grounded in a curated pinout reference to reduce hallucinated pin assignments.

## Supported boards & runtimes

| Board | Runtime(s) |
|---|---|
| ESP32 | Arduino C++, MicroPython, CircuitPython |
| Raspberry Pi (Linux) | Python (RPi.GPIO / smbus / gpiozero) |
| Raspberry Pi Pico (RP2040) | MicroPython, CircuitPython |
| Arduino Uno | Arduino C++ |
| ESP8266 (NodeMCU) | Arduino C++ |
| Teensy 4.0 | Arduino C++ |
| M5Stack Core2 | Arduino C++ |

## Benchmark results

Automated evaluation on a 12-question held-out set (never seen during training), deterministic decoding:

| Metric | 1.5B baseline (65 examples) | **7B (373 examples)** |
|---|---|---|
| Format compliance | 58% | **83%** |
| Runtime contamination-free* | 83% | **100%** |
| Code syntax valid | 92% | **~100%**† |
| Generation speed (RTX 4070 8GB laptop) | 13.7 tok/s | 12.0 tok/s |
| Generation speed (data-center GPU) | — | 73.2 tok/s |
| Adapter size | 81 MB | 165 MB |

\* *Contamination = wrong-runtime API leaking in, e.g. MicroPython's `machine.Pin` appearing in Arduino code, or Raspberry Pi's pin-numbering convention appearing in ESP32 answers. This was the primary failure mode this project set out to fix.*
† *Two apparent syntax failures in the raw eval were confirmed to be generation-length truncation (900-token cap cutting off otherwise-correct code), not real errors — see `eval_report_7B.json`.*

We also tested a LoRA rank-32 variant against the shipped rank-16 adapter: **no measurable quality difference** on any metric, so we ship the smaller, more efficient rank-16 adapter.

## ⚠️ Known limitations — read before using

This is a small-dataset (373 examples) v0.1 release. Manual testing found a clear capability boundary:

**Reliable** — straightforward single-component wiring for boards/parts resembling the training set (e.g. "wire a BME280 to an ESP32 over I2C", "read a DHT22 on a Raspberry Pi"). Three independent manual tests in this category all produced correct, usable answers.

**Unreliable** — genuinely novel multi-component combinations or protocol-level troubleshooting outside the training distribution. Manual testing found three distinct hallucination patterns on such questions:
1. Internal inconsistencies within a single answer (e.g. a wiring diagram wire that contradicts the accompanying table)
2. Fabricated low-level procedures for problems the model wasn't trained on (e.g. an invented, non-functional "software address remapping" procedure for an I2C address conflict)
3. Invented capabilities not present in the requested hardware (e.g. a fabricated "CO2 estimate" from a sensor that cannot measure CO2)

**Always verify wiring against the component's actual datasheet before connecting real hardware.** Incorrect voltage or pin assignments can damage components. Treat every generated answer as a first draft, not a final instruction set.

## Quickstart

```bash
pip install unsloth
```

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.load_adapter("EzioDEVio/arduino-embedded-qwen2.5-coder-7b")
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "How do I wire a BME280 to an ESP32 over I2C?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=1300, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

Minimum hardware: 8GB VRAM GPU (tested on an RTX 4070 laptop GPU at 12 tokens/sec). A 0.5B distilled variant is planned for CPU-only / Raspberry Pi Zero 2W deployment via GGUF.

## Repository contents

| File | Purpose |
|---|---|
| `full_dataset.jsonl` | 373 training examples (wiring table + ASCII diagram + code + explanation format) |
| `pinouts.json` | Curated pinout reference for all 7 supported boards, used to ground generation |
| `task_list.json` | The 260 board+component+runtime task specifications used to generate the dataset |
| `generate_dataset.py` | Synthetic data generation script (calls the Anthropic API, grounded on `pinouts.json`) |
| `validate_dataset.py` | Automated validation gate — checks format compliance and runtime contamination before any example is merged into training data |
| `train_lora.py` | LoRA fine-tuning script (Unsloth) |
| `evaluate_model.py` | Automated evaluation harness (format, contamination, syntax, speed, VRAM) |
| `eval_dataset.jsonl` | The 12-question held-out evaluation set |
| `ask_model.py` | Interactive CLI for asking the model questions directly |

## Reproducing / extending this

The full pipeline is included, not just the weights. To add a new board or component:
1. Add pinout data to `pinouts.json`
2. Add task entries to `task_list.json`
3. Run `python generate_dataset.py` (requires an `ANTHROPIC_API_KEY`)
4. Run `python validate_dataset.py generated_dataset.jsonl` and review rejections
5. Merge into `full_dataset.jsonl` and re-run `train_lora.py`

## Contributing

Found a wrong pin mapping or a bad answer? Please open an issue with the question, the incorrect output, and the correct information if you know it — corrections are the fastest way to improve the next training batch. See `CONTRIBUTING.md`.

## License

Apache 2.0 (see `LICENSE`), inherited from the base model, [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) (Apache 2.0, Alibaba Qwen team).

## Acknowledgments

- [Qwen team](https://github.com/QwenLM/Qwen2.5-Coder) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for the fine-tuning framework
