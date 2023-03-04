from dataclasses import dataclass, field
from datasets import load_from_disk
from typing import Optional
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
import nltk

nltk.download("punkt")
from nltk import sent_tokenize


def postprocess_text(preds, labels, metric='rouge'):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    if metric == 'rouge':
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
    elif metric == 'bleu':
        labels = [[label] for label in labels]
    return preds, labels


@dataclass
class ModelParams:
    model_name_or_path: str = field(
        default="KETI-AIR/ke-t5-base", metadata={"help": "모델의 경로 혹은 이름을 입력합니다."}
    )

    model_auth_token: Optional[str] = field(
        default=None, metadata={"help": "모델의 인증 토큰을 입력합니다."}
    )

    max_seq_length: int = field(
        default=512, metadata={"help": "입력 시퀀스의 최대 길이를 설정합니다."}
    )

    metric_name: str = field(
        default="bleu", metadata={"help": "입력 시퀀스의 최대 길이를 설정합니다."}
    )


@dataclass
class DataParams:
    data_name_or_path: str = field(
        default="qa-dataset", metadata={"help": "데이터의 경로 혹은 이름을 입력합니다."}
    )

    data_auth_token: Optional[str] = field(
        default=None, metadata={"help": "데이터의 인증 토큰을 입력합니다."}
    )

    text_column: str = field(
        default="text",  metadata={"help": "입력 텍스트의 컬럼 이름을 입력합니다."}
    )

    query_column: str = field(
        default="question", metadata={"help": "입력 텍스트의 컬럼 이름을 입력합니다."}
    )

    train_split: str = field(
        default="train", metadata={"help": "학습 데이터의 스플릿 이름을 입력합니다."}
    )

    eval_split: str = field(
        default="validation", metadata={"help": "평가 데이터의 스플릿 이름을 입력합니다."}
    )


@dataclass
class TrainParams(TrainingArguments):
    output_dir: str = field(
        default="output_dir", metadata={"help": "저장할 경로를 입력합니다."}
    )


def main():
    parser = HfArgumentParser((ModelParams, DataParams, TrainParams))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        use_auth_token=model_args.model_auth_token,
    )
    dataset = load_from_disk(
        data_args.data_name_or_path,
    )

    def example_function(examples):
        # tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            examples[data_args.text_column],
            examples[data_args.query_column],
            max_length=model_args.max_seq_length,
            padding="max_length",
            truncation="only_first"
        )

        if "label" in examples:
            # tokenizer.padding_side = "right"
            tokenized_inputs['labels'] = tokenizer(
                examples['label'],
                max_length=model_args.max_seq_length,
                padding=True,
                truncation=True
            )["input_ids"]
        return tokenized_inputs

    dataset = dataset.map(
        example_function,
        batched=True,
        remove_columns=dataset[data_args.train_split].column_names
    )
    print(dataset)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    _metric = evaluate.load(*model_args.metric_name.split("-"))

    def compute_metrics(p):
        preds, labels = p
        preds = preds[0] if isinstance(preds, tuple) else preds
        decoded_preds = tokenizer.batch_decode(np.argmax(preds, axis=-1), skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels, model_args.metric_name
        )

        result = _metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True if model_args.metric_name == 'rouge' else False
        )

        result = {key: value.mid.fmeasure * 100
                  for key, value in result.items()} \
            if model_args.metric_name == 'rouge' else result

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    if train_args.do_train:
        trainer.train()

    elif train_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()