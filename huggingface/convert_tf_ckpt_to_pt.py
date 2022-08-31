import os
import argparse
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers import TFT5ForConditionalGeneration

# [TODO] Model name mapping of calling clasee dictionary.

def loading(args):

    config = T5Config.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5ForConditionalGeneration.from_pretrained(
            f"{args.path}", from_tf=args.from_tf, config=config
    )

    model_tf = TFT5ForConditionalGeneration.from_pretrained(
            f"{args.path}", config=config
    )
    return config, tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_tf", action='store_true', default=False)
    parser.add_argument("--path", type=str,)
    # parser.add_argument("--tokenizer_name", type=str, required=False)
    # parser.add_argument("--config_name", type=str, required=False)
    parser.add_argument("--save_to_path", type=str, default=None)
    parser.add_argument("--share_to_name", type=str, default=None)
    args = parser.parse_args()

    # loading
    model = {}
    model['config'], model['tokenizer'], model['checkpoint'] = loading(args)

    # saving
    if args.save_to_path:
        model['checkpoint'].save_pretrained(args.save_to_path)
        model['tokenizer'].save_pretrained(args.save_to_path)

    # pushing
    if args.share_to_name:
        model['checkpoint'].push_to_hub(args.share_to_name, use_temp_dir=True)
        model['tokenizer'].push_to_hub(f'https://huggingface.co/{args.share_to_name}')
