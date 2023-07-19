import csv
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import evaluate
import numpy as np
import torch
from datasets import Audio
from datasets import DatasetDict
from datasets import load_dataset
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer

MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000


def filter_data(input_length, labels_length):
    """Filter inputs with zero input length or longer than 30s"""
    if not 0 < input_length < max_input_length and labels_length < 448:
        print("Filter one data!")
    return 0 < input_length < max_input_length and labels_length < 448


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["content_segment_audio_path"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(audio)
    batch["labels"] = tokenizer(
        batch["content_segment_normalized_text"], ).input_ids
    batch["labels_length"] = len(batch["labels"])
    return batch


dataset = load_dataset("voidful/NMSQA_audio")
dataset = dataset.remove_columns([
    "id",
    "title",
    "context",
    "content_audio_sampling_rate",
    "content_segment_text",
    "question",
    "answers",
    "content_full_audio_path",
    "content_audio_speaker",
    "question_audio_path",
    "question_audio_sampling_rate",
    "question_audio_speaker",
    "question_normalized_text",
])
dataset = dataset.cast_column("content_segment_audio_path",
                              Audio(sampling_rate=16000))
print(dataset)
processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en", language="English", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-medium.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium.en", language="English", task="transcribe")
filtered_dataset = dataset.filter(
    lambda example: example["content_segment_audio_path"] is not None)
print(filtered_dataset)
filtered_dataset = filtered_dataset.map(
    prepare_dataset,
    writer_batch_size=2048,
    batch_size=32,
    load_from_cache_file=True,
    cache_file_names={
        "train": "nmsqa-train",
        "dev": "nmsqa-dev",
        "test": ""
    },
)

filtered_dataset = filtered_dataset.filter(
    filter_data, input_columns=["input_length", "labels_length"])

print(filtered_dataset)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{
            "input_features": feature["input_features"]
        } for feature in features]
        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id
            ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# model.config.max_length = 768
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-ft/medium",  # change to a repo name of your choice
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    warmup_steps=500,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    generation_max_length=225,
    predict_with_generate=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=filtered_dataset["train"],
    eval_dataset=filtered_dataset["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()
