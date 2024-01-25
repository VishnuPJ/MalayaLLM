# MalayaLLM [മലയാളം/Malayalam]
- A 7B Llama-2 Indic model.
- Continually LoRA PreTrained and FineTuned on “Malayalam” token.


1) Download the dataset

	* Go to Data folder.
	* Download all the file in the link "https://huggingface.co/datasets/uonlp/CulturaX/tree/main/ml" to a folder "data/CulturaX".
    * Run "parquet2txt.py" .It will create a file called "data_clm.txt".
	* Download "https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/ml.tar.xz" and unzip it.
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

		* For finetuning I am using translated alpaca dataset(English to Malayalam). [https://huggingface.co/datasets/tatsu-lab/alpaca]
		* Run "translate_alpaca_dataset.py".This will create "translated_eng2mlm.csv" which will contain the translated version of Alpaca dataset.

   * Finetune the model on huggingface "VishnuPJ/Alpaca_Instruct_Malayalam" dataset.

		* Run "finetune.py".
		* Finetuned LORA weights will be saved to "output_finetune" folder.


5) Inference

	* If you want you can merge the finetuned LORA  weights in "output_finetune" folder with the MalayaLLM pretrained weight in "merged_lora_llama_pretrained_hf" folder using "merge_lora_with_llama.py".
	* Otherwise we will load both the weight files and merge while inferencing.
	* Run "infer.py" for inferencing. Change the instuction to generate the response.
	* You can use "https://varnamproject.com/editor/#/" to transliterate from Manglish to Malyalam.


6) Generate .GGUF model

	* Refer the link [https://www.substratus.ai/blog/converting-hf-model-gguf-model/]

7) Push to hub.

	* Run "Utils\push2hub.py".
