# unit-to-unit evaluation
import json
import os
from asrp.code2voice_model.hubert import hifigan_hubert_layer6_code100
from asrp.code2voice_model.mhubert import hifigan_mhubert_en_layer11_code1000
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
import argparse
import soundfile as sf




# Argument parsing for command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help='Huggingface model name or local model saving position')
parser.add_argument("--pred_valid_json_name", type=str, default='hubert_dev_pred.json', help='Output JSON name for validation predictions')
parser.add_argument("--pred_train_json_name", type=str, default='hubert_train_pred.json', help='Output JSON name for training predictions')
parser.add_argument("--dataset", type=str, default='GSQA/speech-alpaca-gpt4-unit', help='Huggingface dataset name')
parser.add_argument("--save_ans_audio", action="store_true", help='Flag to save answer audios')
parser.add_argument("--audio_save_path", type=str, default='./ans-audio', help='Path to save answer audios')
parser.add_argument("--auto_split_dataset", action="store_true", help='Flag to automatically split dataset into train and test')
parser.add_argument("--pred_trainset", action="store_true", default=False, help='Flag to predict on training set')
parser.add_argument("--valid_set", type=str, default='dev', help='Subset name for validation set in Huggingface dataset')
config = parser.parse_args()




# Function to initialize and load Whisper model and processor
def initialize_whisper():
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
    whisper_model.config.forced_decoder_ids = None
    return whisper_processor, whisper_model

# Function to initialize and load tokenizer and Seq2Seq model
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model).to("cuda")
    return tokenizer, model

# Function to load and split dataset
def load_and_split_dataset():
    dataset = load_dataset(config.dataset)
    if config.auto_split_dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        return dataset['train'], dataset['test']
    else:
        return dataset['train'], dataset[config.valid_set]




# Function to process input and generate transcriptions
def process_and_generate(input_unit, cs, tokenizer, model, whisper_processor, whisper_model):
    # Generating code from input unit
    inputs = tokenizer("".join([f"v_tok_{i}" for i in input_unit]), return_tensors="pt").to("cuda")
    code = tokenizer.batch_decode(model.generate(**inputs, max_length=1024, do_sample=True, top_p=0.85))[0]
    code = [int(i) for i in code.replace("</s>", "").replace("<s>", "").split("v_tok_")[1:]]
    
    # Generating audio from code and transcribing
    audio = cs(code)
    if config.save_ans_audio:
        audio_file_name = f'{config.audio_save_path}/{len(os.listdir(config.audio_save_path)) + 1}.wav'
        sf.write(audio_file_name, audio, 16000)
    
    input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription




# Main processing function
def process_dataset(dataset, dataset_type):
    output = []
    whisper_processor, whisper_model = initialize_whisper()
    tokenizer, model = initialize_model()
    
    for qa_item in dataset:
        cs = hifigan_hubert_layer6_code100()
        ans_dict = {}

        try:
            if "alpaca" in config.dataset:
                question_unit = json.loads(qa_item["hubert_layer6_code100_input_code"])[0]["merged_code"]
                answer_unit = json.loads(qa_item["hubert_layer6_code100_output_audio"])[0]["merged_code"]
            else:
                question_unit = json.loads(qa_item["hubert_100_question_unit"])[0]["merged_code"]
                context_unit = json.loads(qa_item["hubert_100_context_unit"])[0]["merged_code"]
                answer_unit = json.loads(qa_item["hubert_100_answer_unit"])[0]["merged_code"]
        except:
            continue

        groundtruth_answer = cs(answer_unit)
        input_features = whisper_processor(groundtruth_answer, sampling_rate=16000, return_tensors="pt")["input_features"].to("cuda")
        predicted_ids = whisper_model.generate(input_features)
        groundtruth_transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        if "alpaca" in config.dataset:
            pred_transcription = process_and_generate(question_unit, cs, tokenizer, model, whisper_processor, whisper_model)
        else:
            pred_transcription = process_and_generate(question_unit + context_unit, cs, tokenizer, model, whisper_processor, whisper_model)

        ans_dict["gt"] = groundtruth_transcription
        ans_dict["pred"] = pred_transcription
        output.append(ans_dict)

    with open(config.pred_valid_json_name if dataset_type == 'valid' else config.pred_train_json_name, "w") as f:
        json.dump(output, f, indent=4)

# Creating directory for audio save path if it doesn't exist
if config.save_ans_audio:
    try:
        os.makedirs(config.audio_save_path)
    except:
        pass

# Processing the datasets
train_dataset, valid_dataset = load_and_split_dataset()
process_dataset(valid_dataset, 'valid')

if config.pred_trainset:
    process_dataset(train_dataset, 'train')
