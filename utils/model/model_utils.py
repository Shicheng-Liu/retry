# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
#from transformers.deepspeed import HfDeepSpeedConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig


from .reward_model import RewardModel
from ..utils import load_state_dict_into_model


def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    ds_config=None,
    rlhf_training=False,
    disable_dropout=False,
):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            trust_remote_code=True,
            # use_flash_attention_2=True if "llama" in model_name_or_path else False,
            config=model_config,
        )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    return model


def create_critic_model(
    model_name_or_path,
    tokenizer,
    ds_config,
    num_padding_at_beginning=0,
    rlhf_training=False,
    disable_dropout=False,
    zero_stage=0,
    eval_mode=False,
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = RewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model
