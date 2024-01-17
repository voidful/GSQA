import torch
# alpaca text-guided-unit
def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("GSQA/speech-alpaca-gpt4-unit")
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    valid_dataset = dataset['test']

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts
        q, a = batch['hubert_layer6_code100_input_code'],batch['hubert_layer6_code100_output_audio']
        v_tok_q, v_tok_a = convert_vtok(q), convert_vtok(a)
        inputs = tokenizer(v_tok_q, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        answer_texts = batch["output"]

        # Tokenize answers and create labels
        # answer_texts = [i["output"] for i in batch["answers"]]
        text_labels = tokenizer(answer_texts, padding=True, truncation=True, return_tensors="pt").input_ids[0]
        text_labels= [text_labels[:-1]]
        text_labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in text_labels]

        unit_labels = tokenizer(v_tok_a, padding=True, truncation=True, return_tensors="pt").input_ids
        unit_labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in unit_labels]
        # 32099: <extra_id_0>
        all_labels = [text_labels[0]+[torch.tensor(32099)]+unit_labels[0]]
        # print(all_labels)

        assert len(input_ids) == len(unit_labels)
        # with open("test.json", "w") as test_file:
        #     json.dump({
        #         "v_tok_q": v_tok_q,
        #         "v_tok_c": v_tok_c,
        #         "v_tok_a": v_tok_a,
        #         "input_ids": input_ids.tolist(), 
        #         "attention_mask": attention_mask.tolist(), 
        #         "labels": labels.tolist()
        #         }, 
        #         test_file, 
        #         indent=4)
        # raise
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": all_labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        cache_file_name="hubert_train_alpaca",
        load_from_cache_file=True
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size,
        cache_file_name="hubert_valid_alpaca",
        load_from_cache_file=True
    )

    train_dataset = train_dataset.filter(lambda example: len(example["labels"])<4096)
    valid_dataset = valid_dataset.filter(lambda example: len(example["labels"])<4096)

    return train_dataset, valid_dataset

import json
# Check the mismatch between (question, context) and (answer). 
def convert_vtok(unit_code):
    for i in range(len(unit_code)):
        try:
            code = json.loads(unit_code[i])[0]['merged_code']
        except:
            continue
        v_tok = [f"v_tok_{unit}" for unit in code]
        unit_code[i] = ' '.join(v_tok) # blank is not needed
    return unit_code

# def convert_vtok(batch):
#     q, c, a = batch['hubert_100_question_unit'], batch['hubert_100_context_unit'], batch['hubert_100_answer_unit']
#     assert len(q) == len(c) == len(a)
#     for i in range(len(q)):
#         try:
#             unit_q, unit_c, unit_a = json.loads(q[i])[0]['merged_code'], json.loads(c[i])[0]['merged_code'], json.loads(a[i])[0]['merged_code']
#         except:
#             unit_q, unit_c, unit_a = "", "", ""
#             continue
#         v_tok_q = [f"v_tok_{unit}" for unit in unit_q]
#         v_tok_c = [f"v_tok_{unit}" for unit in unit_c]
#         v_tok_a = [f"v_tok_{unit}" for unit in unit_a]
#         q[i] = ' '.join(v_tok_q)
#         c[i] = ' '.join(v_tok_c)
#         a[i] = ' '.join(v_tok_a)
#     return q, c, a

if __name__ == "__main__":
    import math
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments
    )

    # Load model and tokenizer and Set training parameters
    tokenizer = AutoTokenizer.from_pretrained("voidful/long-t5-encodec-tglobal-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("voidful/long-t5-encodec-tglobal-base")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./training_output/Hubert",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        learning_rate=5e-5,
        bf16=True,
        gradient_accumulation_steps=4,
    )
    # Define a data collator to handle tokenization
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # Load dataset
    train_dataset, valid_dataset = get_train_valid_dataset(training_args, tokenizer, model.config)
    print(train_dataset)