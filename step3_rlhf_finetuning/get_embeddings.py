# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from tqdm import tqdm
from utils.model.model_utils import create_critic_model
from utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_reward",
        type=str,
        help="Path to reward model",
        required=True,
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to test prompts",
        required=True,
    )
    parser.add_argument(
        "--unsatisfactory_data_path",
        type=str,
        help="Path to test prompts",
        required=True,
    )
    parser.add_argument(
        "--data_name",
        type=str,
        help="data name",
        required=True,
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        help="output path",
        required=True,
    )
    parser.add_argument(
        "--unsatisfactory_output_path",
        type=str,
        help="output path",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")


    args = parser.parse_args()

    return args


def load_stuff(model_name_or_path, num_padding_at_beginning,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path,
                                tokenizer,
                                None,
                                num_padding_at_beginning,
                                rlhf_training=True,
                                disable_dropout=True,
                                eval_mode=True)

    return model, tokenizer

# def prepare_singlesample(prompt,
#                          good_ans,
#                          tokenizer,
#                          max_seq_len=512,
#                          end_of_conversation_token="<|endoftext|>"):
#     chosen_sentence = prompt + good_ans + end_of_conversation_token
#     chosen_token = tokenizer(chosen_sentence,
#                              max_length=max_seq_len,
#                              padding="max_length",
#                              truncation=True,
#                              return_tensors="pt")

#     batch = {}
#     batch["input_ids"] = chosen_token["input_ids"]
#     batch["attention_mask"] = chosen_token["attention_mask"]

#     return batch

# def opt_embedding(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning):
#     batch = prepare_singlesample(prompt, response, reward_tokenizer, max_seq_len=512, end_of_conversation_token=end_of_conversation_token)
#     batch = to_device(batch, device)
#     reward_model.eval()
#         # Run inference
#     with torch.no_grad():
#         outputs = reward_model.get_embedding(**batch, prompt_length=max(2, num_padding_at_beginning))
#         return outputs
       
# def get_embedding(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning,reward_name):
#     if "opt" in reward_name:
#         embedding = opt_embedding(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning)
#     return embedding

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len):
    chosen_sentence = prompt + good_ans
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch
       
def get_embedding(prompt,response,reward_model,reward_tokenizer,device,max_seq_len,num_padding_at_beginning):
    batch = prepare_singlesample(prompt, response, reward_tokenizer, max_seq_len=max_seq_len)
    batch = to_device(batch, device)
    reward_model.eval()
    with torch.no_grad():
        embedding = reward_model.get_embedding(**batch, prompt_length=max(2, num_padding_at_beginning))
    return embedding


def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    reward_model, reward_tokenizer = load_stuff(args.model_name_or_path_reward,
                                    args.num_padding_at_beginning,
                                    additional_special_tokens)
    reward_model.to(device)
    

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    train_ds = load_dataset("json", data_files=args.train_data_path)["train"]
    train_embedding_list =[]

    unsatisfactory_ds = load_dataset("json", data_files=args.unsatisfactory_data_path)["train"]
    unsatisfactory_embedding_list = []

    train_prompts = train_ds["prompt"]
    train_chosen = train_ds["chosen"]
    train_rejected = train_ds["rejected"]
    train_length = len(train_prompts)
    train_start = int(train_length * 0.2)
    train_end = int(train_length * 0.6)
    train_prompts = train_prompts[train_start:train_end]
    train_chosen = train_chosen[train_start:train_end]
    train_rejected = train_rejected[train_start:train_end]
    for train_prompt, train_good_ans, train_bad_ans in tqdm(zip(train_prompts, train_chosen, train_rejected), total=len(train_prompts),desc="Getting training embeddings"):
        train_chosen_embedding = get_embedding(train_prompt, train_good_ans, reward_model,reward_tokenizer, device,args.max_seq_len,args.num_padding_at_beginning)
        train_rejected_embedding = get_embedding(train_prompt, train_bad_ans, reward_model,reward_tokenizer, device,args.max_seq_len,args.num_padding_at_beginning)
        train_embedding = train_chosen_embedding - train_rejected_embedding
        train_embedding_list.append({"index": len(train_embedding_list), "embedding": train_embedding})

    torch.save(train_embedding_list, f"{args.train_output_path}")

    unsatisfactory_prompts = unsatisfactory_ds["prompt"]  
    unsatisfactory_responses = unsatisfactory_ds["response"] 
    for unsatisfactory_prompt, unsatisfactory_response in tqdm(zip(unsatisfactory_prompts, unsatisfactory_responses),total=len(unsatisfactory_prompts),desc="Getting unsatisfactory embeddings"):

        unsatisfactory_embedding = get_embedding(unsatisfactory_prompt,unsatisfactory_response,reward_model,reward_tokenizer,device,args.max_seq_len,args.num_padding_at_beginning)
        unsatisfactory_embedding_list.append({"index": len(unsatisfactory_embedding_list), "embedding": unsatisfactory_embedding})    
    torch.save(unsatisfactory_embedding_list, f"{args.unsatisfactory_output_path}")


if __name__ == "__main__":
    main()
