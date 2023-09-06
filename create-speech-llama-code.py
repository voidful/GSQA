from datasets import load_dataset, Audio
from asrp.voice2code_model.hubert import hubert_layer6_code100
from asrp.voice2code_model.mhubert import mhubert_layer11_code1000

dataset = load_dataset("voidful/speech-alpaca-gpt4")
dataset = dataset.cast_column("input_audio", Audio(sampling_rate=16000))
dataset = dataset.cast_column("output_audio", Audio(sampling_rate=16000))

hubert = hubert_layer6_code100(batch=5)
mhubert = mhubert_layer11_code1000(batch=5)

import json
import torch
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def handel_data(item):
    with torch.no_grad():
        item['mhubert_layer11_code1000_input_code'] = json.dumps(
            mhubert(input_values=torch.tensor(item['input_audio']['array'], dtype=torch.float32), feat_norm=False,
                    beamsearch=False, top_k=100, beamsize=5), cls=NpEncoder)
        item['mhubert_layer11_code1000_output_audio'] = json.dumps(
            mhubert(input_values=torch.tensor(item['output_audio']['array'], dtype=torch.float32), feat_norm=False,
                    beamsearch=False, top_k=100, beamsize=5), cls=NpEncoder)
        item['hubert_layer6_code100_input_code'] = json.dumps(
            hubert(input_values=torch.tensor(item['input_audio']['array'], dtype=torch.float32), feat_norm=False,
                   beamsearch=False, top_k=100, beamsize=5), cls=NpEncoder)
        item['hubert_layer6_code100_output_audio'] = json.dumps(
            hubert(input_values=torch.tensor(item['output_audio']['array'], dtype=torch.float32), feat_norm=False,
                   beamsearch=False, top_k=100, beamsize=5), cls=NpEncoder)

    return item

dataset.map(handel_data, batched=False)