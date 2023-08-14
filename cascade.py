import json
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import pipeline
from datasets import Audio
# load model and processor
whisper_processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v2", language="English", task="transcribe")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2").to("cuda")
whisper_model.config.forced_decoder_ids = None
whisper_model.config.suppress_tokens = []

def get_audio_data(dataset, mode):
    dataset = dataset[mode]
    dataset = dataset.remove_columns([
        "id",
        "title",
        "context",
        "content_audio_sampling_rate",
        "content_segment_text",
        "question",
        "content_audio_speaker",
        "content_segment_audio_path",
        "question_audio_sampling_rate",
        "question_audio_speaker",
        "question_normalized_text",
    ])
    dataset = dataset.cast_column([
            "content_full_audio_path",
            "question_audio_path",
        ], Audio(sampling_rate=16000)
    )

    return dataset


def process_and_generate(input_audio):
    input_features = whisper_processor(
        input_audio, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription



dataset = load_dataset("voidful/NMSQA-CODE")
audio_dataset = get_audio_data(dataset, "dev")
answer_pred, answer_gt = [], []
for item in audio_dataset:
    context_audio = item['context_full_audio_path']
    question_audio = item['question_audio_path']
    ground_truth = item['answers']['text']
    context = process_and_generate(context_audio)
    question = process_and_generate(question_audio)
    question_answerer = pipeline("question-answering", model='') # Our fine-tuned LongT5 model on SQuADv2(text) 
    answer_pred.append(question_answerer(question=question, context=context)['answer'])
    answer_gt.append(ground_truth)

all_pred = []
for i in range(len(answer_pred)):
    d = dict()
    d['gt'] = answer_gt[i]
    d['pred'] = answer_pred[i]
    all_pred.append(d)
with open("pred.json", "w") as f:
    json.dump(all_pred, f, indent=4)

