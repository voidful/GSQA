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
