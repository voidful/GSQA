import json # $ export HF_DATASETS_CACHE="/../../work/u1210625/.cache/huggingface/datasets"
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import pipeline
from datasets import Audio
from pydub import AudioSegment

# TODO: install pkg following the direction of the below website
# https://learn.microsoft.com/zh-tw/azure/ai-services/speech-service/get-started-text-to-speech?tabs=macos%2Cterminal&pivots=programming-language-python
# TODO: export your env variable in terminal first
# TTS cascade API
# export SPEECH_KEY=
# export SPEECH_REGION=eastus

import os
import azure.cognitiveservices.speech as speechsdk
# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
speech_config.speech_synthesis_language = "en-US" 
# result = speech_synthesizer.speak_text_async("I'm excited to try text to speech").get()
# stream = speechsdk.AudioDataStream(result)
# stream.save_to_wav_file("file.wav")


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
        "question_voice",
        "answer_voice",
        "passage_voice",
    ])

    dataset.cast_column("passage_audio", Audio(sampling_rate=16000))
    dataset.cast_column("question_audio", Audio(sampling_rate=16000))
    dataset.cast_column("answer_audio", Audio(sampling_rate=16000))

    return dataset


# ASR
def process_and_generate(input_audio):
    input_features = whisper_processor(
        input_audio, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription


# tts audio file saving dir name
tts_audio_dir_name = "narrativeQA_audio"
os.mkdir(tts_audio_dir_name)


dataset = load_dataset("voidful/narrativeqa-test-tts")
audio_dataset = get_audio_data(dataset, "train")
answer_pred, answer_gt = [], []
# print(audio_dataset[0])
i = 0 
for item in audio_dataset:
    context_audio = item['passage_audio']
    question_audio = item['question_audio']
    if context_audio==None or question_audio==None:
        continue
    ground_truth = item['answer']
    context = process_and_generate(context_audio["array"])
    question = process_and_generate(question_audio["array"])
    
    # QA LM
    question_answerer = pipeline("text2text-generation", model="MaggiePai/long-t5-encodec-tglobal-base-drop-narrativeQA-squad2-newsQA-superGlue")
    pred_QA_text = question_answerer("question: "+question+" context: "+context)[0]["generated_text"]
    # print(pred_QA_text)
    
    # QA text to speech 
    if pred_QA_text != [] and pred_QA_text != "." and pred_QA_text != "": 
        # TTS
        result = speech_synthesizer.speak_text_async(pred_QA_text).get()
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(f"{tts_audio_dir_name}/file{i}.wav")
        syn_audio = AudioSegment.from_file(f"{tts_audio_dir_name}/file{i}.wav").set_frame_rate(16000)
         # ASR
        # print(syn_audio.get_array_of_samples())
        pred_text = process_and_generate(syn_audio.get_array_of_samples())
        i += 1

    # un-tts-able answer
    else:
        pred_text = ""
    # print(pred_text)
    
    answer_pred.append(pred_text)
    answer_gt.append(ground_truth)


all_pred = []
for i in range(len(answer_pred)):
    d = dict()
    d['gt'] = answer_gt[i]
    d['pred'] = answer_pred[i]
    all_pred.append(d)


with open("pred_cascade_narrativeQA-test.json", "w") as f:
    json.dump(all_pred, f, indent=4)

