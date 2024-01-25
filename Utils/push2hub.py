import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import SFTTrainer


from huggingface_hub import login
login("HF_TOKEN") # With write permission


model_name = "merged_lora_base_pretrained_huggingface"

new_model = "output_finetune"


print(f"Loading model...")
# Load base model

#This will upload the MalayaLLM 7B Base model

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Uncomment the following two lines to upload the finetuned MalaLLM 7B Instruct model

# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


base_model.push_to_hub("MalayaLLM_7B_Base")
tokenizer.push_to_hub("MalayaLLM_7B_Base")

# Uncomment the following two lines to upload the finetuned MalaLLM 7B Instruct model

# base_model.push_to_hub("MalayaLLM_7B_Instruct_v0.1")
# tokenizer.push_to_hub("MalayaLLM_7B_Instruct_v0.1")