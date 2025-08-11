# # Copyright (c) Microsoft Corporation.
# # SPDX-License-Identifier: Apache-2.0

# # DeepSpeed Team
# import argparse
# import logging
# import torch
# import json
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM
# import sys
# import os
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# )

# from utils.model.model_utils import create_hf_model
# from utils.utils import to_device, load_hf_tokenizer
# from deepspeed import get_accelerator
# from datasets import load_dataset

# logger = logging.getLogger(__name__)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
#     parser.add_argument(
#         "--model_name_or_path_baseline",
#         type=str,
#         help="Path to baseline model",
#         required=True,
#     )
#     parser.add_argument(
#         "--model_name_or_path_finetune",
#         type=str,
#         help="Path to pretrained model",
#         required=True,
#     )
#     parser.add_argument(
#         "--model_name_or_path_rlhf",
#         type=str,
#         help="Path to rlhf model",
#         required=True,
#     )
#     parser.add_argument(
#         "--model_name_or_path_xrlhf",
#         type=str,
#         help="Path to xrlhf model",
#         required=True,
#     )
#     parser.add_argument(
#         "--output_path",
#         type=str,
#         help="Path to output file",
#         required=True,
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         help="model name",
#         required=True,
#     )
#     parser.add_argument(
#         "--data_name",
#         type=str,
#         help="data name",
#         required=True,
#     )
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         help="Path to test prompts",
#         required=True,
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help='batch size',
#     )
#     parser.add_argument(
#         "--num_padding_at_beginning",
#         type=int,
#         default=1,
#         help=
#         "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
#         "We did not see this in other models but keep it as an option for now.",
#     )
#     parser.add_argument(
#         "--num_beams",
#         type=int,
#         default=1,
#         help='Specify num of beams',
#     )
#     parser.add_argument(
#         "--num_beam_groups",
#         type=int,
#         default=1,
#         help='Specify num of beams',
#     )
#     parser.add_argument(
#         "--top_k",
#         type=int,
#         default=4,
#         help='Specify num of beams',
#     )
#     parser.add_argument(
#         "--penalty_alpha",
#         type=float,
#         default=0.6,
#         help='Specify num of beams',
#     )
#     parser.add_argument(
#         "--num_return_sequences",
#         type=int,
#         default=1,
#         help='Specify num of return sequences',
#     )
#     parser.add_argument(
#         "--max_new_tokens",
#         type=int,
#         default=512,
#         help='Specify num of return sequences',
#     )
#     parser.add_argument("--language",
#                         type=str,
#                         default="English",
#                         choices=["English", "Chinese", "Japanese"])
#     parser.add_argument(
#         "--add_eot_token",
#         action='store_true',
#         help="Add <|endoftext|> as additional special token to tokenizer")

#     args = parser.parse_args()

#     return args

# def generate(model,
#              tokenizer,
#              inputs,
#              num_beams=1,
#              num_beam_groups=1,
#              do_sample=False,
#              num_return_sequences=1,
#              max_new_tokens=100):

#     generate_ids = model.generate(inputs.input_ids,
#                                   attention_mask=inputs.attention_mask,
#                                   num_beams=num_beams,
#                                   num_beam_groups=num_beam_groups,
#                                   do_sample=do_sample,
#                                   num_return_sequences=num_return_sequences,
#                                   max_new_tokens=max_new_tokens)

#     result = tokenizer.batch_decode(generate_ids,
#                                     skip_special_tokens=True,
#                                     clean_up_tokenization_spaces=False)
#     return result


# def generate_constrastive_search(model,
#                                  tokenizer,
#                                  inputs,
#                                  top_k=4,
#                                  penalty_alpha=0.6,
#                                  num_return_sequences=1,
#                                  max_new_tokens=100):

#     generate_ids = model.generate(inputs.input_ids,
#                                   top_k=top_k,
#                                   penalty_alpha=penalty_alpha,
#                                   num_return_sequences=num_return_sequences,
#                                   max_new_tokens=max_new_tokens)

#     result = tokenizer.batch_decode(generate_ids,
#                                     skip_special_tokens=True,
#                                     clean_up_tokenization_spaces=False)
#     return result


# def print_utils(gen_output):
#     for i in range(len(gen_output)):
#         print()
#         print(gen_output[i])
#         print()


# def prompt_eval(args, model_fintuned, model_rlhf, model_xrlhf, tokenizer, device, prompts):

#     finetune_response = []
#     rlhf_response = []
#     xrlhf_response = []
#     for prompt in tqdm(prompts, desc="Generating responses"):
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
#         prompt_length = len(prompt)



#         r_finetune_g = generate(model_fintuned,
#                                 tokenizer,
#                                 inputs,
#                                 num_beams=1,
#                                 num_return_sequences=args.num_return_sequences,
#                                 max_new_tokens=args.max_new_tokens)
#         finetune_response.append(r_finetune_g[0][prompt_length:])

#         r_rlhf_g = generate(model_rlhf,
#                             tokenizer,
#                             inputs,
#                             num_beams=1,
#                             num_return_sequences=args.num_return_sequences,
#                             max_new_tokens=args.max_new_tokens)
#         rlhf_response.append(r_rlhf_g[0][prompt_length:])

#         r_xrlhf_g = generate(model_xrlhf,
#                             tokenizer,
#                             inputs,
#                             num_beams=1,
#                             num_return_sequences=args.num_return_sequences,
#                             max_new_tokens=args.max_new_tokens)
#         xrlhf_response.append(r_xrlhf_g[0][prompt_length:])

#     test_results = []
#     for p, s, r, x in zip(prompts, finetune_response, rlhf_response, xrlhf_response):
#         test_results.append({
#             "prompt": p,
#             "response_sft": s,
#             "response_rlhf": r,
#             "response_xrlhf": x
#         })
#     with open(f"{args.output_path}","w") as f:
#         json.dump(test_results,f)

# def main():
#     args = parse_args()

#     device = torch.device(get_accelerator().device_name(0))

#     args.end_of_conversation_token = "<|endoftext|>"
#     #additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
#     tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
#                                   fast_tokenizer=True)  #,add_special_tokens=additional_special_tokens

#     model_fintuned = create_hf_model(AutoModelForCausalLM,
#                                      args.model_name_or_path_finetune,
#                                      tokenizer, None)
#     model_rlhf = create_hf_model(AutoModelForCausalLM,
#                                      args.model_name_or_path_rlhf,
#                                      tokenizer, None)
    
#     model_xrlhf = create_hf_model(AutoModelForCausalLM,
#                                      args.model_name_or_path_xrlhf,
#                                      tokenizer, None)
    
#     model_fintuned.to(device)
#     model_rlhf.to(device)
#     model_xrlhf.to(device)
    
#     ds = load_dataset("json", data_files=args.data_path)["train"]
#     prompts = ds["prompt"]
#     print("enter prompt")
#     prompt_eval(args,  model_fintuned, model_rlhf, model_xrlhf, tokenizer, device, prompts)


# if __name__ == "__main__":
#     main()

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import argparse
import logging
import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import sys
import os
from multiprocessing import Process, Queue
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer
from deepspeed import get_accelerator
from datasets import load_dataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument("--model_name_or_path_baseline", type=str, required=True)
    parser.add_argument("--model_name_or_path_finetune", type=str, required=True)
    parser.add_argument("--model_name_or_path_rlhf", type=str, required=True)
    parser.add_argument("--model_name_or_path_xrlhf", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese", "Japanese"])
    parser.add_argument("--add_eot_token", action='store_true')
    return parser.parse_args()

def model_worker(base_model_path,model_path, device_id, prompts_with_idx, args, model_tag, result_queue):
    torch.cuda.set_device(device_id)
    tokenizer = load_hf_tokenizer(base_model_path, fast_tokenizer=True)

    model = create_hf_model(AutoModelForCausalLM, model_path, tokenizer, None)
    model.to(f"cuda:{device_id}")
    model.eval()

    results = []
    for idx, prompt in tqdm(prompts_with_idx, desc=f"{model_tag} on cuda:{device_id}"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{device_id}")
        prompt_length = len(prompt)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=1,
                num_beam_groups=1,
                do_sample=False,
                num_return_sequences=args.num_return_sequences,
                max_new_tokens=args.max_new_tokens
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        results.append((idx, model_tag, decoded[prompt_length:]))

    result_queue.put(results)

def parallel_prompt_eval(args, indexed_prompts):
    model_paths = {
        #"baseline": args.model_name_or_path_baseline,
        "finetuned": args.model_name_or_path_finetune,
        "rlhf": args.model_name_or_path_rlhf,
        "xrlhf": args.model_name_or_path_xrlhf,
    }

    result_queue = Queue()
    processes = []
    num_gpus = torch.cuda.device_count()

    for i, (tag, model_path) in enumerate(model_paths.items()):
        device_id = i % num_gpus
        p = Process(target=model_worker, args=(args.model_name_or_path_baseline,model_path, device_id, indexed_prompts, args, tag, result_queue))
        p.start()
        processes.append(p)

    all_results = []
    total_expected = len(indexed_prompts) * len(model_paths)
    pbar = tqdm(total=total_expected, desc="Generating")

    for _ in processes:
        results = result_queue.get()
        all_results.extend(results)
        pbar.update(len(results))

    for p in processes:
        p.join()

    pbar.close()

    grouped = defaultdict(dict)
    for idx, model_tag, response in all_results:
        grouped[idx][f"response_{model_tag}"] = response

    test_results = []
    for idx, prompt in indexed_prompts:
        test_results.append({
            "prompt": prompt,
            #"response_base": grouped[idx].get("response_baseline", ""),
            "response_sft": grouped[idx].get("response_finetuned", ""),
            "response_rlhf": grouped[idx].get("response_rlhf", ""),
            "response_xrlhf": grouped[idx].get("response_xrlhf", "")
        })

    with open(args.output_path, "w") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    ds = load_dataset("json", data_files=args.data_path)["train"]
    prompts = ds["prompt"][:500]
    indexed_prompts = list(enumerate(prompts))
    parallel_prompt_eval(args, indexed_prompts)

if __name__ == "__main__":
    main()
