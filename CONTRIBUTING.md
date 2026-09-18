# Contributing

This project improves through real-world corrections. Here's how to help.

## Report a wrong answer

Open an issue including:
1. The exact question you asked
2. The full model output
3. What's wrong with it (wrong pin, wrong library, invented capability, etc.)
4. The correct information, if you know it (a datasheet link is great)

Corrections like this go straight into the next training batch via `task_list.json` and `generate_dataset.py`.

## Add a new board or component

1. Add pinout/reference data to `pinouts.json` (see existing entries for the format)
2. Add one or more task entries to `task_list.json`:
   ```json
   {"board": "YourBoard", "components": ["Your Component"], "protocol": "I2C", "language": "cpp"}
   ```
3. Run the generator (requires an `ANTHROPIC_API_KEY`):
   ```bash
   python generate_dataset.py
   ```
4. Validate before merging:
   ```bash
   python validate_dataset.py generated_dataset.jsonl
   ```
5. Review `generated_dataset_rejected.jsonl` — some rejections are real contamination catches, others may be false positives worth a closer look (see the validator's `RUNTIME_SIGNATURES` if you need to tune the checks for a new language/runtime)
6. Open a PR with the new/updated `full_dataset.jsonl`, `pinouts.json`, and `task_list.json`

## Improve the evaluation set

`eval_dataset.jsonl` is intentionally small (12 questions). More held-out questions covering boards/runtimes/components not already in the eval set are welcome — this makes the benchmark numbers more trustworthy.

## Code of conduct

Be kind, be specific, and back claims about hardware behavior with a datasheet or your own tested result where possible.
