import pandas as pd
import googletrans
from googletrans import Translator
from datasets import load_dataset
import concurrent.futures
import subprocess

def translate_text(index, text):
    print(" {} of {} completed".format(index, len(text)))
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='ml')
    return translated_text.text

def main():
    dataset = load_dataset("tatsu-lab/alpaca")
    mlm_lst = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(translate_text, i, text) for i, text in enumerate(dataset["train"]["text"])]

        for future in concurrent.futures.as_completed(futures):
            mlm_lst.append(future.result())

    df = pd.DataFrame(mlm_lst, columns=['Prompt'])
    print("completed")
    df.to_csv('translated_eng2mlm.csv', encoding="utf-8")

if __name__ == "__main__":
    main()
