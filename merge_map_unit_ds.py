from datasets import DatasetDict, Dataset
from datasets import load_dataset

dataset = load_dataset("voidful/NMSQA")

merged_dataset_dict = DatasetDict()

dataset1 = dataset.load_from_disk("GSQA-hubert/")
dataset2 = dataset.load_from_disk("GSQA-mhubert/")

length = 0

for i in dataset['train']:
    if len(i['context_unit']) == 0:
        length += 1
    elif len(i['question_unit']) == 0:
        length += 1
    elif len(i['answer_unit']) == 0:
        length += 1
print("empty data", length)

merged_ds_train = []
for i, j in zip(dataset1['train'], dataset2['train']):
    if i['id'] == j['id']:
        merged_ds_data = i
        merged_ds_data["hubert_100_context_unit"] = i["context_unit"]
        merged_ds_data["hubert_100_question_unit"] = i["question_unit"]
        merged_ds_data["hubert_100_answer_unit"] = i["answer_unit"]
        merged_ds_data["mhubert_1000_context_unit"] = j["context_unit"]
        merged_ds_data["mhubert_1000_question_unit"] = j["question_unit"]
        merged_ds_data["mhubert_1000_answer_unit"] = j["answer_unit"]
        merged_ds_train.append(merged_ds_data)
    else:
        print("ERROR")

merged_ds_test = []
for i, j in zip(dataset1['test'], dataset2['test']):
    if i['id'] == j['id']:
        merged_ds_data = i
        merged_ds_data["hubert_100_context_unit"] = i["context_unit"]
        merged_ds_data["hubert_100_question_unit"] = i["question_unit"]
        merged_ds_data["hubert_100_answer_unit"] = i["answer_unit"]
        merged_ds_data["mhubert_1000_context_unit"] = j["context_unit"]
        merged_ds_data["mhubert_1000_question_unit"] = j["question_unit"]
        merged_ds_data["mhubert_1000_answer_unit"] = j["answer_unit"]
        merged_ds_test.append(merged_ds_data)
    else:
        print("ERROR")

merged_ds_dev = []
for i, j in zip(dataset1['dev'], dataset2['dev']):
    if i['id'] == j['id']:
        merged_ds_data = i
        merged_ds_data["hubert_100_context_unit"] = i["context_unit"]
        merged_ds_data["hubert_100_question_unit"] = i["question_unit"]
        merged_ds_data["hubert_100_answer_unit"] = i["answer_unit"]
        merged_ds_data["mhubert_1000_context_unit"] = j["context_unit"]
        merged_ds_data["mhubert_1000_question_unit"] = j["question_unit"]
        merged_ds_data["mhubert_1000_answer_unit"] = j["answer_unit"]
        merged_ds_dev.append(merged_ds_data)
    else:
        print("ERROR")

merged_dataset_dict['train'] = Dataset.from_list(merged_ds_train)
merged_dataset_dict['test'] = Dataset.from_list(merged_ds_test)
merged_dataset_dict['dev'] = Dataset.from_list(merged_ds_dev)
merged_dataset_dict = merged_dataset_dict.remove_columns(["context_unit", "question_unit", "answer_unit"])


def remove_field(example):
    del example['answers']['audio_full_neg_answer_end']
    del example['answers']['audio_full_neg_answer_start']
    return example


for split in merged_dataset_dict.keys():
    merged_dataset_dict[split] = merged_dataset_dict[split].map(remove_field)

length = 0
for i in merged_dataset_dict['train']:
    if len(i['hubert_100_context_unit']) == 0:
        length += 1
    elif len(i['hubert_100_question_unit']) == 0:
        length += 1
    elif len(i['hubert_100_answer_unit']) == 0:
        length += 1
print("empty data", length)
