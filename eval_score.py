import json
import os

from datasets import load_dataset


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


def exact_match(pred, gt):
    correct = 0
    assert len(pred) == len(gt)
    for i in range(len(pred)):
        pred[i] = normalize_text(pred[i])
        gt[i] = normalize_text(gt[i])
    for i in range(len(pred)):
        correct += pred[i] == gt[i]
    return 100 * correct / len(pred)


def compute_f1_score(pred, gt):
    f1 = 0
    assert len(pred) == len(gt)
    for i in range(len(pred)):
        if not isinstance(pred[i], list):
            pred[i] = normalize_text(pred[i]).split()
        if not isinstance(gt[i], list):
            gt[i] = normalize_text(gt[i]).split()
        common_token = set(pred[i]) & set(gt[i])
        if len(common_token) == 0:
            f1 += 0
            continue
        prec = len(common_token) / len(pred[i])
        rec = len(common_token) / len(gt[i])
        f1 += 2 * prec * rec / (prec + rec)
    return 100 * f1 / len(pred)


dataset = load_dataset("voidful/NMSQA-CODE")
train_set, dev_set = dataset["train"], dataset["dev"]
train_text_gt, dev_text_gt = [], []

for i in range(len(train_set)):
    if len(train_text_gt) == 10000:
        break
    if train_set[i]["hubert_100_answer_unit"] != "":
        train_text_gt.append(train_set[i]["answers"]["text"][0])

for i in range(len(dev_set)):
    if dev_set[i]["hubert_100_answer_unit"] != "":
        dev_text_gt.append(dev_set[i]["answers"]["text"][0])

hubert_train_pred, hubert_train_gt, hubert_dev_pred, hubert_dev_gt = [], [], [], []
with open(os.path.join("hubert_train_pred.json"), "r") as f:
    all_train = json.load(f)
    for item in all_train:
        hubert_train_pred.append(item["pred"])
        hubert_train_gt.append(item["gt"])
with open(os.path.join("hubert_dev_pred.json"), "r") as f:
    all_dev = json.load(f)
    for item in all_dev:
        hubert_dev_pred.append(item["pred"])
        hubert_dev_gt.append(item["gt"])

print("HuBERT Unit EM on dev set: ", exact_match(hubert_dev_pred,
                                                 hubert_dev_gt))
print("HuBERT Unit EM on train set: ",
      exact_match(hubert_train_pred, hubert_train_gt))

print("HuBERT Text EM on dev set: ", exact_match(hubert_dev_pred, dev_text_gt))
print("HuBERT Text EM on train set: ",
      exact_match(hubert_train_pred, train_text_gt))

print("HuBERT GT EM on dev set: ", exact_match(hubert_dev_gt, dev_text_gt))
print("HuBERT GT EM on train set: ", exact_match(hubert_train_gt,
                                                 train_text_gt))

print(
    "HuBERT Unit F1-score on dev set: ",
    compute_f1_score(hubert_dev_pred, hubert_dev_gt),
)
print(
    "HuBERT Unit F1-score on train set: ",
    compute_f1_score(hubert_train_pred, hubert_train_gt),
)

print("HuBERT Text F1-score on dev set: ",
      compute_f1_score(hubert_dev_pred, dev_text_gt))
print(
    "HuBERT Text F1-score on train set: ",
    compute_f1_score(hubert_train_pred, train_text_gt),
)

print("HuBERT GT F1-score on dev set: ",
      compute_f1_score(hubert_dev_gt, dev_text_gt))
print(
    "HuBERT GT F1-score on train set: ",
    compute_f1_score(hubert_train_gt, train_text_gt),
)

# mhubert_train_pred, mhubert_train_gt, mhubert_dev_pred, mhubert_dev_gt = [], [], [], []
# with open(os.path.join("train_dev_pred", "mhubert_train_pred.json"), "r") as f:
#     all_train = json.load(f)
#     for item in all_train:
#         mhubert_train_pred.append(item['pred'])
#         mhubert_train_gt.append(item['gt'])
# with open(os.path.join("train_dev_pred", "mhubert_dev_pred.json"), "r") as f:
#     all_dev = json.load(f)
#     for item in all_dev:
#         mhubert_dev_pred.append(item['pred'])
#         mhubert_dev_gt.append(item['gt'])

# print("mHuBERT Unit EM on dev set: ", exact_match(mhubert_dev_pred, mhubert_dev_gt))
# print("mHuBERT Unit EM on train set: ", exact_match(mhubert_train_pred, mhubert_train_gt))

# print("mHuBERT Text EM on dev set: ", exact_match(mhubert_dev_pred, dev_text_gt))
# print("mHuBERT Text EM on train set: ", exact_match(mhubert_train_pred, train_text_gt))

# print("mHuBERT GT EM on dev set: ", exact_match(mhubert_dev_gt, dev_text_gt))
# print("mHuBERT GT EM on train set: ", exact_match(mhubert_train_gt, train_text_gt))

# print("mHuBERT Unit F1-score on dev set: ", compute_f1_score(mhubert_dev_pred, mhubert_dev_gt))
# print("mHuBERT Unit F1-score on train set: ", compute_f1_score(mhubert_train_pred, mhubert_train_gt))

# print("mHuBERT Text F1-score on dev set: ", compute_f1_score(mhubert_dev_pred, dev_text_gt))
# print("mHuBERT Text F1-score on train set: ", compute_f1_score(mhubert_train_pred, train_text_gt))

# print("mHuBERT GT F1-score on dev set: ", compute_f1_score(mhubert_dev_gt, dev_text_gt))
# print("mHuBERT GT F1-score on train set: ", compute_f1_score(mhubert_train_gt, train_text_gt))
