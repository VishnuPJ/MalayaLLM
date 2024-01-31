# MalayaLLM [മലയാളം/Malayalam]
<img src="Assets/Baby_MalayaLLM.png" alt="MalayaLLM Image" width="300" height="auto">

- A 7B LLaMA-2 Indic model.
- Continually LoRA PreTrained and FineTuned on “Malayalam” tokens.

This is an attempt to construct a Language Model (LLM) focused on **generative AI for Malayalam language**. While several LLMs are proficient in supporting multiple languages, including Malayalam, enhancing their performance for specific tasks such as content generation and question answering specifically in Malayalam can be achieved through dedicated training on a Malayalam dataset. In pursuit of this, I've undertaken the **continuous pre-training of the LLAMA2 model using a comprehensive Malayalam dataset**.

The model is currently in its early stages, and ongoing training and fine-tuning with a more comprehensive dataset are necessary to enhance its performance. I will consistently provide updated revisions to the model.

# Model description
The MalayaLLM models have been improved and customized to incorporate a comprehensive Malayalam vocabulary comprising approximately 18,000 tokens, expanding upon the groundwork laid by the original LLaMA-2.

- **Model type:** A 7B LLaMA2 pretrained model on Malayalam .
- **Language(s):** Malayalam and English
- **Source Model:** [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Training Precision:** `float16`

## Datasets Used
  * For Pretraining/Tokenization
	* [ai4bharat](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/ml.tar.xz)
 	* [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX/tree/main/ml)
  * For Finetuning
  	* [Alpaca_Instruct_Malayalam](https://huggingface.co/datasets/VishnuPJ/Alpaca_Instruct_Malayalam)

 ## Available Models

| Model                    | Type                        | Data              | Base Model           | # Params | Download Links                                                         |
|--------------------------|-----------------------------|-------------------|----------------------|------|------------------------------------------------------------------------|
| MalayaLLM 7B Base   #v0.1   | Base model                  | 12GB              | LLaMA 7B             | 7B   | [HF Hub](https://huggingface.co/VishnuPJ/MalayaLLM_7B_Base)     |
| MalayaLLM 7B Instruct  #v0.1| Instruction following model | 52k instructions | MalayaLLM 7B Base  | 7B   | [HF Hub](https://huggingface.co/VishnuPJ/MalayaLLM_7B_Instruct_v0.1) |
| MalayaLLM 7B Instruct  #v0.2| Instruction following model | 52k instructions | MalayaLLM 7B Base  | 7B   | [HF Hub](https://huggingface.co/VishnuPJ/MalayaLLM_7B_Instruct_v0.2) |

### Quantized Version of Available Models

| Model                    | Format | Bits                 | Download Links                                                               |
|--------------------------|--------|----------------------|------------------------------------------------------------------------------|
| MalayaLLM 7B Instruct   #v0.1  | GGUF   | Q8_0 | [HF Hub](https://huggingface.co/VishnuPJ/MalayaLLM_7B_Instruct_v0.1_GGUF)      |
| MalayaLLM 7B Instruct   #v0.2  | GGUF   | Q8_0 | [HF Hub](https://huggingface.co/VishnuPJ/MalayaLLM_7B_Instruct_v0.2_GGUF)      |

## A simple example code

```python
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

model_name = "VishnuPJ/MalayaLLM_7B_Instruct_v0.2"
print(f"Loading model...")
# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=200)
sys_prompt = "ഒരു ടാസ്ക് വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക."

while True:
    inst = input("Enter instruction (or 'exit' to end): ")
    if inst.lower() == 'exit':
        break
    # Generate response using the user-provided instruction
    result = pipe(f"{sys_prompt} ### Instruction: {inst} ### Response:")
    # Print the generated text
    print(result[0]['generated_text'])
```

## Example Output
```
Enter instruction (or 'exit' to end): സൂര്യൻ ഉദിക്കുന്ന ദിശ ഏതെന്നു പറയുക .
ഒരു ടാസ്ക് വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### Instruction: സൂര്യൻ ഉദിക്കുന്ന ദിശ ഏതെന്നു പറയുക . ### Response: സൂര്യൻ ഉദിക്കുന്ന ദിശ കിഴക്കായിരിക്കും.
Enter instruction (or 'exit' to end): Where does the Sun rise?
ഒരു ടാസ്ക് വിവരിക്കുന്ന ഒരു നിർദ്ദേശം ചുവടെയുണ്ട്. അഭ്യർത്ഥന ശരിയായി പൂർത്തിയാക്കുന്ന ഒരു പ്രതികരണം എഴുതുക. ### Instruction: Where does the Sun rise? ### Response: The Sun rises in the east.
Enter instruction (or 'exit' to end): exit
```

## Getting Started

### Steps to run pretraining and finetuning
1) Download the dataset

	* Go to Data folder.
	* Download all the file in the link "[CulturaX](https://huggingface.co/datasets/uonlp/CulturaX/tree/main/ml)" to a folder "data/CulturaX".
    * Run "parquet2txt.py" .It will create a file called "data_clm.txt".
	* Download "[ai4bharat](https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/ml.tar.xz)" and unzip it.
	* Copy "data_clm.txt" and "ml.txt" to a folder "data/ml". 

2) Tokenization
	
   * Create Malayalam token files.

		* Go to Tokenizer folder.
		* Run "tokenizer.py".This will create malayalam tokens from the text file(s) provided.(You can also give the path to a single ".txt" file also)
		* It will create two files "MalayaLLM.model" and "MalayaLLM.vocab".

   * Merge Malayalam tokens with 32k LLama2 tokens.

		* Run "merge_tokenizer.py".It will generate two folders "merged_tokenizer_sentencepiece" and "merged_tokenizer_huggingface".
		* "merged_tokenizer_huggingface" will contain the merged tokenizer.

3) Pretrain

   * Download base Llama-2 model.

		* Go to Pretrain folder.
		* Run "download_base_llama.py". This will download Llam2 7B base model to "llama_7b_hf" folder.

   * Pretrain.

		* Create two folders "output_dir" and "cache_dir".
		* Modify "run_pt.sh".
		* Run "./run_pt.sh". (This will start the pretraining and checkpoints will be saved to "output_dir".) 
		* If you want to resume training from checkpoints , comment "--overwrite_output_dir" in "run_pt.sh" and run again.

   * Merge Pretrained LORA weights with Base Llama2 weights.

		* Run the command,
		```bash
			python merge_lora_with_llama.py \
    		--base_model path/to/llama/model \  # llama_7b_hf
    		--lora_model path/to/first/lora/model [path/to/second/lora/model] \ # checkpoint-22500
    		--output_type [pth|huggingface] \ # huggingface
    		--output_dir path/to/output/dir  # merged_lora_llama_pretrained_hf
        ```
    	* This will merge the base Llama2 and pretrained LORA weights into a folder "merged_lora_llama_pretrained_hf"

4) Finetune

   * Translate Alpaca instruct dataset to malayalam.

		* For finetuning I am using translated alpaca dataset(English to Malayalam).
 [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
		* Run "translate_alpaca_dataset.py".This will create "translated_eng2mlm.csv" which will contain the translated version of Alpaca dataset.

   * Finetune the model on huggingface "VishnuPJ/Alpaca_Instruct_Malayalam" dataset.

		* Run "finetune.py".
		* Finetuned LORA weights will be saved to "output_finetune" folder.


5) Inference

	* If you want you can merge the finetuned LORA  weights in "output_finetune" folder with the MalayaLLM pretrained weight in "merged_lora_llama_pretrained_hf" folder using "merge_lora_with_llama.py".
	* Otherwise we will load both the weight files and merge while inferencing.
	* Run "infer.py" for inferencing. Change the instuction to generate the response.
	* You can use "[Transliterate](https://www.google.com/intl/ml/inputtools/try/)" to transliterate from Manglish to Malayalam.


6) Generate .GGUF model

	* Refer the link [hf-gguf](https://www.substratus.ai/blog/converting-hf-model-gguf-model/)

7) Push to hub.

	* Run "Utils\push2hub.py".

### Reference
	* [Continual Pre-training of Language Models](https://arxiv.org/abs/2302.03241)
 	* [Llama 2](https://arxiv.org/abs/2307.09288)
  	* [Chinese-LLaMA](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main)
	* [tamil-llama](https://github.com/abhinand5/tamil-llama/blob/main)
