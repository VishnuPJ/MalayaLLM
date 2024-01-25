import sentencepiece as spm

'''
### Dataset used
# !wget https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/ml.tar.xz

### Unzip it
# !tar -xvf /content/ml.tar.xz

### ml.txt contains total of 56061611 lines
# !wc -l /content/data/ml/ml.txt

### Splitting the text into managable chunks
# !split -l 7007701 /content/data/ml/ml.txt /content/data/ml/ml_half.txt
'''

### Training the tokenizer using SentencePiece , bpe(byte pair encoding)
spm.SentencePieceTrainer.train(
            input="data/ml/ml.txt",
            model_prefix="MalayaLLM",
            vocab_size=20000,
            character_coverage=1.0,
            model_type="bpe",
        )

### To load and test the tokenizer
# sp = spm.SentencePieceProcessor()
# sp.load('MalayaLLM.model')
# print(sp.encode_as_pieces('അത് പരീക്ഷിച്ചുനോക്കുന്നതിന്'))