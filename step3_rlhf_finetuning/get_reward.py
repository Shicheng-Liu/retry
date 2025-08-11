from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from deepspeed import get_accelerator
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.model.model_utils import create_critic_model
from utils.utils import to_device, load_hf_tokenizer


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

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
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

def opt_reward(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning):
    batch = prepare_singlesample(prompt, response, reward_tokenizer, max_seq_len=2048, end_of_conversation_token=end_of_conversation_token)
    batch = to_device(batch, device)
    reward_model.eval()
        # Run inference
    with torch.no_grad():
        outputs = reward_model.forward_value(**batch, prompt_length=max(2, num_padding_at_beginning))
        return outputs["chosen_end_scores"].item()

def OpenAssistant_reward(prompt,response,reward_model,reward_tokenizer,device):
    input_ids = reward_tokenizer(prompt,response,return_tensors='pt')
    input_ids.to(device)
    reward = reward_model(**input_ids).logits[0].cpu().detach()
    return reward

def PKU_reward(prompt,response,reward_model,reward_tokenizer,device):
    input = prompt + response
    input_ids = reward_tokenizer(input,return_tensors='pt',truncation=True,max_length=2048)
    input_ids.to(device)
    output = reward_model(**input_ids)
    return output.end_scores.item()
       
def get_reward(prompt,response,reward_model,reward_tokenizer,device,reward_name):
    if "opt" in reward_name:
        reward = opt_reward(prompt,response,reward_model,reward_tokenizer,device,None,1)
    if "PKU" in reward_name:
        reward = PKU_reward(prompt,response,reward_model,reward_tokenizer,device)
    if "OpenAssistant" in reward_name:
        reward = OpenAssistant_reward(prompt,response,reward_model,reward_tokenizer,device)
    return reward

def main():
    #reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    reward_model_name = "/efs/shicheng/remax/step4_xrlhf_finetuning/output/opt-1.3b/full-hh-rlhf/unlearned_reward"


    device = torch.device(get_accelerator().device_name(0))

    if "opt" in reward_model_name:
        reward_model, reward_tokenizer = load_stuff(reward_model_name,
                                        1,
                                        None)

    if "OpenAssistant" in reward_model_name:
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    if "PKU-Alignment" in reward_model_name:
        from safe_rlhf.models import AutoModelForScore
        reward_model = AutoModelForScore.from_pretrained(reward_model_name, torch_dtype=torch.bfloat16, device_map='auto')
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    reward_model.to(device)

    ds = load_dataset("allenai/reward-bench")['filtered']
    prompts = ds['prompt']
    chosen = ds['chosen']
    rejected = ds['rejected']
    agreement_list =[]

    for prompt, chose, reject in tqdm(zip(prompts, chosen, rejected),total=len(prompts),desc="Evaluation process"):
        chosen_reward = get_reward(prompt,chose,reward_model,reward_tokenizer,device,reward_model_name)
        rejected_reward = get_reward(prompt,reject,reward_model,reward_tokenizer,device,reward_model_name)
        if chosen_reward > rejected_reward:
            agreement_list.append(1)
        else:
            agreement_list.append(0)
    print(sum(agreement_list)/len(agreement_list))


if __name__ == "__main__":
    main()