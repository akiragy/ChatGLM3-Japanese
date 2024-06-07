from constants import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from llamafactory.train.tuner import run_exp, export_model
import yaml


if __name__ == "__main__":
    # train
    args = yaml.safe_load(open('train_config/step6_train_sft.yaml', 'r'))
    args.pop('deepspeed')
    run_exp(args)

    # merge lora
    merge_args = {
        "model_name_or_path": "your_path/step5_pt_merged",
        "adapter_name_or_path": "your_path/step6_sft_lora",
        "template": "chatglm3",
        "finetuning_type": "lora",
        "export_dir": "your_path/chatglm3-japanese",
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    export_model(merge_args)
