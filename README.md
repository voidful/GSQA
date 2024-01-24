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




---


## Multi-Task Training
Datasets
> Unit Datasets: [GSQA/speech-alpaca-gpt4-unit](https://huggingface.co/datasets/GSQA/speech-alpaca-gpt4-unit)
> Speech Datasets [GSQA/spoken-alpaca-gpt4](https://huggingface.co/datasets/GSQA/spoken-alpaca-gpt4)

[Models Hub](https://huggingface.co/GSQA)
> T5-series Model:[long-T5](https://huggingface.co/voidful/long-t5-encodec-tglobal-base/tree/main)
> alpaca-TQA-init T5-series Model: [LongT5-alpaca-TQA](https://huggingface.co/GSQA/LongT5-alpaca-TQA)

### 1. setting
login GSQA authorized huggingface account
```
$ huggingface-cli login
```
login wandb account to record training figures
```
$ wandb login --relogin
```

### 2. training script



```bash=
# select one of the aux_task in choices to fill after --aux_task
$ python3 main_multiTask.py --aux_task qt,at,qu
(choices=['qt,qu', 'qt,at,qu', "qu,at", "at"])
```

<!-- ### step3
Evaluating Script:

```
python3 whisper_evaluate.py 
python3 BertScore_eval.py # Remember to check the name of output files.
``` -->
### 3. after finish training, push model to https://huggingface.co/GSQA


---

## Unit-to-unit Evaluation
ASR Model:[Whisper]() --> TBD

<!-- Language Model:[Long-T5-HuBERT-Unit](https://huggingface.co/Oscarshih/long-t5-base-SQA), [Long-T5-mHuBERT-Unit](https://huggingface.co/voidful/long-t5-base-SQA-mhubert-1000) -->

Evaluating Script:
```
# stpe1: run
python3 whisper_evaluate.py --model /path/to/the/huggingface/model --auto_split_dataset
# (for more optional arguments check whisper_evaluate.py)

# step 2: for alpaca dataset BertScore, run
python3 BertScore_eval.py
# (remember to change the evaluation file path first)

# step 2: for dataset with context, run
python3 eval_score.py # Remember to check the name of output files.
# Note: Please put the best reported score to Overleaf Table.
