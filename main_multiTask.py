import math
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from torch import nn
import torch
from module.multiTask_data_processing import get_train_valid_dataset
from module.eval_metric import compute_metrics_fn
from transformers import logging
import os
from accelerate import notebook_launcher


# logging.set_verbosity_warning()
# Load model and tokenizer and Set training parameters
import wandb
wandb.init(project="longt5-alpaca-TQA-multiTask")

# v1 TQA model
# tokenizer = AutoTokenizer.from_pretrained("GSQA/longT5-TQA")
# model = AutoModelForSeq2SeqLM.from_pretrained("GSQA/longT5-TQA")


# alpaca TQA pretraining model
tokenizer = AutoTokenizer.from_pretrained("GSQA/LongT5-alpaca-TQA")
model = AutoModelForSeq2SeqLM.from_pretrained("GSQA/LongT5-alpaca-TQA")



training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output/longt5-alpaca-multiTask-unit",
    num_train_epochs=30,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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
    eval_accumulation_steps=8,
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# Load dataset
train_dataset, valid_dataset = get_train_valid_dataset(training_args, tokenizer, model.config)
# print(train_dataset[0])


def compute_metrics_middle_fn(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_metrics_fn(decoded_preds, decoded_labels)

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)





class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, auxiliary_train_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary_train_dataset = auxiliary_train_dataset
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        # labels = inputs.pop("labels")
        labels = inputs["labels"]
        # print(labels)
        auxiliary_inputs = next(iter(self.auxiliary_train_dataset))
        aux_str_inputs = auxiliary_inputs["aux_inputs"]
        aux_inputs = tokenizer(aux_str_inputs, padding=True, truncation=True, return_tensors="pt")
        aux_inputs["decoder_input_ids"] = labels
        aux_inputs["labels"] = labels
        aux_inputs = aux_inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # forward pass
        outputs = model(**inputs)
        aux_outputs = model(**aux_inputs)
        # print(aux_outputs.loss)
        # print(aux_outputs.get("logits").shape)
        
        main_loss = outputs.loss
        aux_loss = aux_outputs.loss
        loss = main_loss + aux_loss
        # print(loss)
        return (loss, outputs) if return_outputs else loss
    


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    auxiliary_train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_middle_fn,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)




# Start training
trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
