import json
import os
from asrp.code2voice_model.hubert import hifigan_hubert_layer6_code100
from asrp.code2voice_model.mhubert import hifigan_mhubert_en_layer11_code1000
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
import soundfile as sf

# load model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
whisper_model.config.forced_decoder_ids = None

tokenizer = AutoTokenizer.from_pretrained("GSQA/longT5-alpaca-text-guide-unit")
model = AutoModelForSeq2SeqLM.from_pretrained("GSQA/longT5-alpaca-text-guide-unit").to("cuda")

dataset = load_dataset("GSQA/speech-alpaca-gpt4-unit")
dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
valid_dataset = dataset['test']

audio_dir = "./longt5-text-guided_ans-audio"

try:
    os.makedirs(audio_dir)
except:
    pass



def process_and_generate(input_unit, cs, n):
    inputs = tokenizer("".join([f"v_tok_{i}" for i in input_unit]),
                       return_tensors="pt").to("cuda")
    code = tokenizer.batch_decode(model.generate(**inputs, max_length=512, do_sample=True, top_p=0.85))[0] #max_length=1024
    # print(code)
    if '<extra_id_0>' in code:
        code = code.split('<extra_id_0>')[1]
        guided_text = code.split('<extra_id_0>')[0]
    else: 
        return "Invalid: without output vtok", None
    
    code = [
        int(i) for i in code.replace("</s>", "").replace("<s>", "").split(
            "v_tok_")[1:]
    ]
    # unit --TTS-- text --ASR-- gt_transcription
    audio = cs(code)
    sf.write(f'{audio_dir}/file_{n}.wav', audio, 16000)
    
    input_features = whisper_processor(
        audio, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0].strip()
    # print(transcription)
    return transcription, guided_text
    


count = 0
output = []
n = 0
ref_sentences = []  # To store groundtruth_transcription sentences
hyp_sentences = []  
for qa_item in valid_dataset:
    cs = hifigan_hubert_layer6_code100()
    ans_dict = {}
    try:
        question_unit = json.loads(
            qa_item["hubert_layer6_code100_input_code"])[0]["merged_code"]
        answer_unit = json.loads(
            qa_item["hubert_layer6_code100_output_audio"])[0]["merged_code"]
    except:
        count += 1
        continue
    # unit --TTS-- text --ASR-- gt_transcription
    groundtruth_answer = cs(answer_unit) # tensor
    input_features = whisper_processor(
        groundtruth_answer, sampling_rate=16000,
        return_tensors="pt")["input_features"].to("cuda")
    predicted_ids = whisper_model.generate(input_features)
    groundtruth_transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    
    
    pred_transcription, guided_text = process_and_generate(question_unit, cs, n)
    if pred_transcription=="Invalid: without output vtok":
        count += 1
        continue
    n += 1
    
    # Append groundtruth_transcription to ref_sentences list
    ref_sentences.append(groundtruth_transcription)
    hyp_sentences.append(pred_transcription)
    
    ans_dict["question"] = qa_item["speech_input"]
    ans_dict["gt"] = groundtruth_transcription
    ans_dict["guided_text"] = guided_text
    ans_dict["pred"] = pred_transcription
    
    output.append(ans_dict)
    
    # if n>=50:
    #     break
    # print(output[-1])
    

with open("alpacaDev_longt5_textGuided.json", "w") as f:
    json.dump(output, f, indent=4)
print("dev_set invalid count: ", count)


# # Write groundtruth_transcription sentences to ref.txt
# with open("ref.txt", "w") as ref_file:
#     for sentence in ref_sentences:
#         ref_file.write(sentence + "\n")
# # Write pred_transcription sentences to hyp.txt
# with open("hyp.txt", "w") as ref_file:
#     for sentence in hyp_sentences:
#         ref_file.write(sentence + "\n")


