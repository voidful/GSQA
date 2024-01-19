from evaluate import load
import json
import os


def normalize_text(s):
    import re
    import string

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def bert_scorer(pred, gt):
    assert len(pred) == len(gt)
    for i in range(len(pred)):
        pred[i] = normalize_text(pred[i])
        gt[i] = normalize_text(gt[i])
        
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=pred, references=gt, lang="en", model_type="microsoft/deberta-xlarge-mnli")
        
    return results



hubert_dev_pred, hubert_dev_gt = [], []
with open(os.path.join("alpacaDev_longt5_textGuided.json"), "r") as f:
    all_dev = json.load(f)
    for item in all_dev:
        hubert_dev_pred.append(item["pred"])
        hubert_dev_gt.append(item["gt"])
        
bert_scores = bert_scorer(hubert_dev_pred, hubert_dev_gt)

precision = sum(bert_scores["precision"])/len(bert_scores["precision"])
recall = sum(bert_scores["recall"])/len(bert_scores["recall"])
f1 = sum(bert_scores["f1"])/len(bert_scores["f1"])


print("BertScore on Alpaca-devset:")
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)
