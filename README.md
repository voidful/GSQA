# GSQA

## Environment Settings
```
pip3 install -r requirements.txt
# pip3 install -r requirements_2.txt # Oscar's local env settings
```


## Fine-tuned LM List
HuBERT Unit:[long-t5-base-SQA-hubert-100](https://huggingface.co/Oscarshih/long-t5-base-SQA)  
mHuBERT Unit:[long-t5-base-SQA-mhubert-1000](https://huggingface.co/voidful/long-t5-base-SQA-mhubert-1000)  


## Training
Datasets: [NMSQA](https://huggingface.co/datasets/voidful/NMSQA-CODE)

T5-series Model:[long-T5](https://huggingface.co/voidful/long-t5-encodec-tglobal-base/tree/main)

<!-- LLaMA Model:[LLaMA v2]() -->

Training Script:
```bash=
python3 main.py
```

## Evaluating
ASR Model:[Whisper]() --> TBD

Language Model:[Long-T5-HuBERT-Unit](https://huggingface.co/Oscarshih/long-t5-base-SQA), [Long-T5-mHuBERT-Unit](https://huggingface.co/voidful/long-t5-base-SQA-mhubert-1000)

Evaluating Script:
```
python3 whisper_evaluate.py 
python3 eval_score.py # Remember to check the name of output files.
# Note: Please put the best reported score to Overleaf Table.
```


## Multi-Task Training
Datasets: [Alpaca](https://huggingface.co/datasets/GSQA/speech-alpaca-gpt4-unit)

T5-series Model:[long-T5](https://huggingface.co/voidful/long-t5-encodec-tglobal-base/tree/main)
alpaca-TQA-init T5-series Model: [LongT5-alpaca-TQA](https://huggingface.co/GSQA/LongT5-alpaca-TQA)

### step0
login GSQA authorized huggingface account
```
$ huggingface-cli login
```
login wandb account to record training figures
```
$ wandb login --relogin
```

### step1
Modify preprocessing script: `module/multiTask_data_processing.py`

To choose different multiTask input, uncomment one of the following aux_str_inputs: 
```
# # task 1. qt,qu
# aux_str_inputs = [ qt+" "+tok_q for qt, tok_q in zip(q_ts, v_tok_q)]
# # task 2. qt,at,qu
# aux_str_inputs = [ qt+" "+at+" "+tok_q for qt, at, tok_q in zip(q_ts, a_ts, v_tok_q)]
# # task 3. qu,at
# aux_str_inputs = [ at+" "+tok_q for at, tok_q in zip(a_ts, v_tok_q)]
# # task 4. at
# aux_str_inputs = [ at for at in a_ts ]
```

### step2
Run Training Script:
```bash=
$ python3 main_multiTask.py
