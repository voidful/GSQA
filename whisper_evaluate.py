import json

from asrp.code2voice_model.hubert import hifigan_hubert_layer6_code100
from asrp.code2voice_model.mhubert import hifigan_mhubert_en_layer11_code1000
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

# load model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2").to("cuda")
whisper_model.config.forced_decoder_ids = None

tokenizer = AutoTokenizer.from_pretrained("Oscarshih/long-t5-base-SQA-15ep")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Oscarshih/long-t5-base-SQA-15ep").to("cuda")
dataset = load_dataset("voidful/NMSQA-CODE")


def process_and_generate(input_unit, cs):
    inputs = tokenizer("".join([f"v_tok_{i}" for i in input_unit]),
                       return_tensors="pt").to("cuda")
    code = tokenizer.batch_decode(model.generate(**inputs, max_length=1024))[0]
    code = [
        int(i) for i in code.replace("</s>", "").replace("<s>", "").split(
            "v_tok_")[1:]
    ]
    audio = cs(code)
    input_features = whisper_processor(
        audio, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription

output = []
for qa_item in dataset["dev"]:
    cs = hifigan_hubert_layer6_code100()
    ans_dict = {}
    try:
        question_unit = json.loads(
            qa_item["hubert_100_question_unit"])[0]["merged_code"]
        context_unit = json.loads(
            qa_item["hubert_100_context_unit"])[0]["merged_code"]
        answer_unit = json.loads(
            qa_item["hubert_100_answer_unit"])[0]["merged_code"]
    except:
        continue
    groundtruth_answer = cs(answer_unit)
    input_features = whisper_processor(
        groundtruth_answer, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    groundtruth_transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    pred_transcription = process_and_generate(question_unit + context_unit, cs)
    ans_dict["gt"] = groundtruth_transcription
    ans_dict["pred"] = pred_transcription
    output.append(ans_dict)
    # print(output[-1])
with open("hubert_dev_pred.json", "w") as f:
    json.dump(output, f, indent=4)

output = []
for qa_item in dataset["train"]:
    if count == 10000:
        break
    cs = hifigan_hubert_layer6_code100()
    ans_dict = {}
    try:
        question_unit = json.loads(
            qa_item["hubert_100_question_unit"])[0]["merged_code"]
        context_unit = json.loads(
            qa_item["hubert_100_context_unit"])[0]["merged_code"]
        answer_unit = json.loads(
            qa_item["hubert_100_answer_unit"])[0]["merged_code"]
    except:
        continue
    groundtruth_answer = cs(answer_unit)
    input_features = whisper_processor(
        groundtruth_answer, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    groundtruth_transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    pred_transcription = process_and_generate(question_unit + context_unit, cs)
    # pred_transcription = process_and_generate(answer_unit)
    ans_dict["gt"] = groundtruth_transcription
    ans_dict["pred"] = pred_transcription
    output.append(ans_dict)

with open("hubert_train_pred.json", "w") as f:
    json.dump(output, f, indent=4)
