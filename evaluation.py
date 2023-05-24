import json

from asrp.code2voice_model.hubert import hifigan_hubert_layer6_code100
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# load model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to('cuda')
whisper_model.config.forced_decoder_ids = None

tokenizer = AutoTokenizer.from_pretrained("Oscarshih/long-t5-base-SQA-15ep")
model = AutoModelForSeq2SeqLM.from_pretrained("Oscarshih/long-t5-base-SQA-15ep").to('cuda')
dataset = load_dataset("voidful/NMSQA-CODE")
cs = hifigan_hubert_layer6_code100()


def process_and_generate(input_unit):
    inputs = tokenizer("".join([f"v_tok_{i}" for i in input_unit]), return_tensors="pt").to('cuda')
    code = tokenizer.batch_decode(model.generate(**inputs, max_length=1024))[0]
    code = [int(i) for i in code.replace("</s>", "").replace("<s>", "").split("v_tok_")[1:]]
    audio = cs(code)
    input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")['input_features'].to('cuda')
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription


for qa_item in dataset['dev']:
    question_unit = json.loads(qa_item['hubert_100_question_unit'])[0]["merged_code"]
    context_unit = json.loads(qa_item['hubert_100_context_unit'])[0]["merged_code"]
    answer_unit = json.loads(qa_item['hubert_100_answer_unit'])[0]["merged_code"]

    groundtruth_answer = cs(answer_unit)

    groundtruth_transcription = process_and_generate(question_unit + context_unit)
    pred_transcription = process_and_generate(answer_unit)

    print(groundtruth_transcription, pred_transcription)
