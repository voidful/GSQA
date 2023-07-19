def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("voidful/NMSQA-CODE")
    train_dataset = dataset['train']
    valid_dataset = dataset['dev']
    # valid_dataset = dataset['validation']

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts
        q, c, a = batch['hubert_100_question_unit'], batch['hubert_100_context_unit'], batch['hubert_100_answer_unit']
        v_tok_q, v_tok_c, v_tok_a = convert_vtok(q), convert_vtok(c), convert_vtok(a)
        inputs = tokenizer(v_tok_q, v_tok_c, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Tokenize answers and create labels
        # answer_texts = [i["text"][0] for i in batch["answers"]]
        labels = tokenizer(v_tok_a, padding=True, truncation=True, return_tensors="pt").input_ids
        labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
        assert len(input_ids) == len(labels)
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
            "labels": labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        cache_file_name="hubert_train",
        # load_from_cache_file=True
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size,
        cache_file_name="hubert_valid",
        # load_from_cache_file=True
    )

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

