from datasets import load_dataset

ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
ds_train = ds['train']
ls_ds_train = list(ds_train)
output_dir = "code.json"
import json
with open(output_dir, "w") as f:
    json.dump(ls_ds_train, f, indent=4)