from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name_or_path = "output_dir/checkpoint-11328/"
auth_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model.push_to_hub("psyche/t5-filling", use_auth_token=auth_token)
tokenizer.push_to_hub("psyche/t5-filling", use_auth_token=auth_token)
