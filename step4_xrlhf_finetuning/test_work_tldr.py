import json
import openai
from tqdm import tqdm
import time
import argparse
import os
from datasets import load_dataset

def build_prompt(prompt, summary1, summary2):
    """GPT-4 prompt"""
    return f"""Which of the following summaries does a better job of summarizing the most important points in the given post, without including unimportant or irrelevant details? A good summary is both precise and concise.
Post:
{prompt}
Summary 1:
{summary1}
Response 2:
{summary2}

FIRST provide a one-sentence comparison of the two summaries and explain which you prefer and why. SECOND, on a new line, state only "1" or "2" to indicate your choice. Your response should be in the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"1" or "2">"""

def parse_gpt_response(response):
    """parse GPT-4 response"""
    lines = response.split('\n')
    comparison = None
    choice = None
    
    for line in lines:
        if line.startswith('Comparison:'):
            comparison = line.replace('Comparison:', '').strip()
        elif line.startswith('Preferred:'):
            choice = line.replace('Preferred:', '').strip()
    
    return comparison, choice

def gpt4_compare(prompt, api_key, max_retries=3):
    """use GPT-4 API to evaluate"""
    openai.api_key = api_key
    for _ in range(max_retries):
        try:
            # response = openai.ChatCompletion.create(
            response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                top_p=1,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(5)
    return None

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--loss_type', required=True, help='loss_type')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    args = parser.parse_args()


    #print("--------------------")
    #print("loss_type",args.loss_type)
    print("--------------------")
    path_RLHF="/gpuhome/hbz5148/workspace/siyuan/retry/step3_rlhf_finetuning/opt-1.3b_tldr_test_result.json"
    path_XRLHF="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/opt-1.3b_tldr_test_result.json"

    RLHF_data = load_dataset("json", data_files=path_RLHF)["train"]
    XRLHF_data = load_dataset("json", data_files=path_XRLHF)["train"]
    prompts = RLHF_data["prompt"]
    response_sft = RLHF_data["response_sft"]
    response_rlhf = RLHF_data["response_rlhf"]
    response_xrlhf = XRLHF_data["response_xrlhf"]
    win_rate_list_rlhf_sft = []
    win_rate_list_xrlhf_sft = []
    win_rate_list_xrlhf_rlhf = []


    for prompt, sft_response, rlhf_response, xrlhf_response in tqdm(zip(prompts, response_sft, response_rlhf, response_xrlhf),total=len(prompts),desc="Evaluation process"):
            
        gpt4_prompt_rlhf_sft = build_prompt(prompt, rlhf_response, sft_response)
        gpt_response_rlhf_sft = gpt4_compare(gpt4_prompt_rlhf_sft, args.api_key)
        if gpt_response_rlhf_sft:
            comparison, choice = parse_gpt_response(gpt_response_rlhf_sft)
            if choice == '1':
                win_rate_list_rlhf_sft.append(1)
            else:
                win_rate_list_rlhf_sft.append(0)
        time.sleep(1) # API speed control

        gpt4_prompt_xrlhf_sft = build_prompt(prompt, xrlhf_response, sft_response)
        gpt_response_xrlhf_sft = gpt4_compare(gpt4_prompt_xrlhf_sft, args.api_key)
        if gpt_response_xrlhf_sft:
            comparison, choice = parse_gpt_response(gpt_response_xrlhf_sft)
            if choice == '1':
                win_rate_list_xrlhf_sft.append(1)
            else:
                win_rate_list_xrlhf_sft.append(0)
        time.sleep(1) # API speed control

        gpt4_prompt_xrlhf_rlhf = build_prompt(prompt, xrlhf_response, rlhf_response)
        gpt_response_xrlhf_rlhf = gpt4_compare(gpt4_prompt_xrlhf_rlhf, args.api_key)
        if gpt_response_xrlhf_rlhf:
            comparison, choice = parse_gpt_response(gpt_response_xrlhf_rlhf)
            if choice == '1':
                win_rate_list_xrlhf_rlhf.append(1)
            else:
                win_rate_list_xrlhf_rlhf.append(0)
        time.sleep(1) # API speed control
           
    print("RLHF_SFT win rate",1.0*sum(win_rate_list_rlhf_sft)/len(win_rate_list_rlhf_sft))
    print("XRLHF_SFT win rate",1.0*sum(win_rate_list_xrlhf_sft)/len(win_rate_list_xrlhf_sft))
    print("XRLHF_RLHF win rate",1.0*sum(win_rate_list_xrlhf_rlhf)/len(win_rate_list_xrlhf_rlhf))

if __name__ == "__main__":
    main()