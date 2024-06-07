from constants import *
import sentencepiece as spm
import pandas as pd
import json

data_path_wiki = '../data_for_tokenizer/wiki_for_spm.txt'
data_path_aozora = '../data_for_tokenizer/aozora_for_spm.txt'

need_to_convert = True
need_to_train = False

if need_to_convert:
    # step1 wiki to text file
    df = pd.read_parquet('../data_for_tokenizer/hf_izumi-lab_wikipedia-ja-20230720_part0.parquet')
    with open(data_path_wiki, 'w', encoding='utf-8') as file:
        for text in df['text']:
            clean_text = text
            file.write(f"{clean_text}\n")
    print("wiki to text done")

    # step2 aozora to text file
    head_article_num = 6000
    with open('../data_for_tokenizer/hf_globis-university_aozorabunko-clean.jsonl', 'r', encoding='utf-8') as infile, open(data_path_aozora, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i == head_article_num:
                break
            data = json.loads(line)
            clean_text = data['text']
            outfile.write(f"{clean_text}\n")
    print("aozora to text done")

if need_to_train:
    # step3 train spm
    train_cmd = f'--input={data_path_wiki},{data_path_aozora} --model_prefix=../data_for_tokenizer/ja_bpe --vocab_size=20000 --character_coverage=0.9995 --model_type=bpe'
    spm.SentencePieceTrainer.Train(train_cmd)

# step4 load and test spm
sp = spm.SentencePieceProcessor(model_file='../data_for_tokenizer/ja_bpe.model')
print(sp.encode_as_pieces('俺たちの上に太陽など無かった。いつも夜。だけど暗くはなかった。太陽に変わるものがあったから。'))