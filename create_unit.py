import argparse
import json
import sys

import numpy as np
import torch
from asrp.voice2code_model.hubert import hubert_layer6_code100
from asrp.voice2code_model.mhubert import mhubert_layer11_code1000
from datasets import load_dataset
from pydub import AudioSegment

dataset = load_dataset("voidful/NMSQA")

ModelMap = {
    'mhubert': mhubert_layer11_code1000,
    'hubert': hubert_layer6_code100,
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ModelMap.keys(), required=True, help="model name")
    input_arg, model_arg = parser.parse_known_args(args)
    input_arg = {k: v for k, v in vars(input_arg).items() if v is not None}
    model_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
    return input_arg, model_arg


def convert_code_fn(batch_data, hc_model, data_set=""):
    context_audio_path = batch_data["content_full_audio_path"]
    question_audio_path = batch_data["question_audio_path"]
    batch_data.update({"context_unit": ''})
    batch_data.update({"question_unit": ''})
    batch_data.update({"answer_unit": ''})

    if context_audio_path is not None:
        context_audio = AudioSegment.from_file(
            f'./NMSQA_audio/{data_set}_audios/{context_audio_path}').set_frame_rate(16000)
    else:
        return batch_data

    if question_audio_path is not None:
        question_audio = AudioSegment.from_file(
            f'./NMSQA_audio/{data_set}_audios/{question_audio_path}').set_frame_rate(16000)
    else:
        return batch_data

    if context_audio is not None and "audio_full_answer_start" in batch_data["answers"] and \
            "audio_full_answer_end" in batch_data["answers"]:
        answer_start = batch_data["answers"]["audio_full_answer_start"][0]
        answer_end = batch_data["answers"]["audio_full_answer_end"][0]
        if len(context_audio[answer_start * 1000:answer_end * 1000]) > 1000:
            answer_audio = context_audio[answer_start * 1000:answer_end * 1000]
        else:
            return batch_data
    else:
        return batch_data

    context_units = hc_model(
        input_values=torch.from_numpy(np.array(context_audio.get_array_of_samples(), dtype=np.float32)),
        feat_norm=False, beamsearch=False, top_k=100, beamsize=5)
    question_units = hc_model(
        input_values=torch.from_numpy(np.array(question_audio.get_array_of_samples(), dtype=np.float32)),
        feat_norm=False, beamsearch=False, top_k=100, beamsize=5)
    answer_units = hc_model(
        input_values=torch.from_numpy(np.array(answer_audio.get_array_of_samples(), dtype=np.float32)),
        feat_norm=False, beamsearch=False, top_k=100, beamsize=5)

    if context_units is not None:
        batch_data.update({"context_unit": json.dumps(context_units, cls=NpEncoder)})

    if question_units is not None:
        batch_data.update({"question_unit": json.dumps(question_units, cls=NpEncoder)})

    if answer_units is not None:
        batch_data.update({"answer_unit": json.dumps(answer_units, cls=NpEncoder)})
    return batch_data


def main(arg=None):
    input_arg, model_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)

    hc_model = ModelMap[input_arg['model']]()
    dataset['test'] = dataset['test'].map(convert_code_fn, batched=False,
                                          fn_kwargs={"data_set": 'test', "hc_model": hc_model})
    dataset['dev'] = dataset['dev'].map(convert_code_fn, batched=False,
                                        fn_kwargs={"data_set": 'dev', "hc_model": hc_model})
    dataset['train'] = dataset['train'].map(convert_code_fn, batched=False,
                                            fn_kwargs={"data_set": 'train', "hc_model": hc_model})
    dataset.save_to_disk(f"GSQA-{input_arg['model']}")


if __name__ == "__main__":
    main()