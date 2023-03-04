from transformers import HfArgumentParser
from dataclasses import dataclass, field
from datasets import load_dataset, DatasetDict, Dataset
from collections import defaultdict
@dataclass
class DataParams:
    data_name_or_path: str = field(
        default="KETI-AIR/korquad > v1.0", metadata={"help":"데이터의 이름 또는 경로를 설정합니다."}
    )


def main():
    parser = HfArgumentParser((DataParams,))
    args = parser.parse_args()
    dataset = load_dataset(*args.data_name_or_path.split(" > "))
    train_outputs = defaultdict(list)
    for d in dataset['train']:
        train_outputs["text"].append(d['context'])
        train_outputs["question"].append(d['question']+" "+"_"*5),
        train_outputs["label"].append(d['question']+" "+d['answers']['text'][0])

    eval_outputs = defaultdict(list)
    for d in dataset['dev']:
        eval_outputs["text"].append(d['context'])
        eval_outputs["question"].append(d['question']+" "+"_"*5),
        eval_outputs["label"].append(d['question']+" "+d['answers']['text'][0])

    dataset = DatasetDict({
        "train": Dataset.from_dict(train_outputs),
        "validation":Dataset.from_dict(eval_outputs)
    })
    print(dataset)
    dataset.save_to_disk("qa-dataset")




if __name__ == "__main__":
    main()
