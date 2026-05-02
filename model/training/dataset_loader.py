from datasets import load_dataset
import json

DATA_PATH = "data/datasets"

def load_py_v_dataset():
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_PATH}/train.jsonl",
            "val": f"{DATA_PATH}/val.jsonl"
        }
    )
    return dataset


def format_example(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    return {"text": prompt}


def get_formatted_dataset():
    dataset = load_py_v_dataset()

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    return dataset