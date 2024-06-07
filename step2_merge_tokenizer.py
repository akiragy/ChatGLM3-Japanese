from constants import *
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from chatglm_origin_code_dir.tokenization_chatglm import ChatGLMTokenizer
import os

# load chinese tokenizer (ChatGLM origin)
cn_tokenizer = ChatGLMTokenizer.from_pretrained(CHATGLM_ORIGIN_CODE_DIR)
print(f'special_tokens_map: {cn_tokenizer.special_tokens_map}')

# load japanese tokenizer (train with sentencepiece)
ja_tokenizer = spm.SentencePieceProcessor(model_file='../data_for_tokenizer/ja_bpe.model')
print(len(cn_tokenizer), len(ja_tokenizer))

# merge pieces
cn_spm = sp_pb2_model.ModelProto()
cn_spm.ParseFromString(cn_tokenizer.tokenizer.sp_model.serialized_model_proto())
ja_spm = sp_pb2_model.ModelProto()
ja_spm.ParseFromString(ja_tokenizer.serialized_model_proto())
cn_spm_tokens_set = set(p.piece for p in cn_spm.pieces)
print(f"Origin model pieces: {len(cn_spm_tokens_set)}")

# add 9 dummy pieces for special tokens
for dummy_piece in ['vcxv1', 'vcxv2', 'vcxv3', 'vcxv4', 'vcxv5', 'vcxv6', 'vcxv7', 'vcxv8', 'vcxv9']:
    if dummy_piece not in cn_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = dummy_piece
        new_p.score = 0
        cn_spm.pieces.append(new_p)
    else:
        assert False

# add ja pieces
for p in ja_spm.pieces:
    piece = p.piece
    if piece not in cn_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        cn_spm.pieces.append(new_p)
print(f"New model pieces: {len(cn_spm.pieces)}")

# save
output_sp_dir = '../data_for_tokenizer/merged_tokenizer_sp'
output_hf_dir = '../data_for_tokenizer/merged_tokenizer_hf'
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir+'/cn_and_ja.model', 'wb') as f:
    f.write(cn_spm.SerializeToString())
tokenizer = ChatGLMTokenizer(vocab_file=output_sp_dir+'/cn_and_ja.model')
tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-Japanese tokenizer has been saved to {output_hf_dir}")

# test
pre_tokenizer = ChatGLMTokenizer.from_pretrained(CHATGLM_ORIGIN_CODE_DIR)
cur_tokenizer = ChatGLMTokenizer.from_pretrained(output_hf_dir)
# test chinese (expected same results)
text = "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"
print("\nTest text (Chinese):\n", text)
print(f"pre tokenizer: {pre_tokenizer.tokenize(text)}")
print(f"cur tokenizer: {cur_tokenizer.tokenize(text)}")
# test japanese (expected shorter results)
text = """どうしてこうなるんだろう…初めて、好きな人が出来た。一生ものの友だちができた。嬉しいことが二つ重なって、
その二つの嬉しさが、また、たくさんの嬉しさを連れてきてくれて。夢のように幸せな時間を手に入れたはずなのに…なのに、どうして、こうなっちゃうんだろう…"""
print("\nTest text (Japenses):\n", text)
print(f"pre tokenizer, len={len(pre_tokenizer.tokenize(text))}: {pre_tokenizer.tokenize(text)}")
print(f"cur tokenizer, len={len(cur_tokenizer.tokenize(text))}: {cur_tokenizer.tokenize(text)}")