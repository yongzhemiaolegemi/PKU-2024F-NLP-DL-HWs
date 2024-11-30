# NLPDL - Assignment 3

## Environment 
This assignment requires only two packages:
- `torch`
- `transformers`

Ensure these are installed in your environment before running the code.

## Code Structure
- `data.txt`: Dataset containing prompts for testing your implementation.
- `main.py`: The main script to run the assignment.
- `customized_gpt2.py`: Contains the code for the customized GPT-2 model.

## Task Overview
Your task is to implement a customized GPT-2 model with KV-cache by inheriting from the `GPT2LMHeadModel` class provided in the `transformers` library.

### Modifiable Sections:
- Implement your own version of `customized_greedy_decoding()` in `main.py`.
- Modify the relevant classes in `customized_gpt2.py`.

The `golden_greedy_decoding_wo_cache()` function in `main.py` serves as the ground truth for testing the correctness of your implementation.

## Bonus (Optional)
For extra credit, you may analyze the patterns in the provided prompt dataset and attempt to optimize the inference speed beyond the default KV-cache technique.
