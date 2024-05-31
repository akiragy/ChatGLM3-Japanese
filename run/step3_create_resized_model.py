from constants import *
import torch
from chatglm_resized_code_dir.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_resized_code_dir.tokenization_chatglm import ChatGLMTokenizer
import shutil, glob

# load resized tokenizer
tokenizer = ChatGLMTokenizer.from_pretrained(CHATGLM_RESIZED_CODE_DIR)

# load original model
model = ChatGLMForConditionalGeneration.from_pretrained(CHATGLM_ORIGIN_MODEL_DIR, device_map='cuda:0', torch_dtype=torch.float16)

# resize model and modify config
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
model.config.padded_vocab_size = model.get_input_embeddings().weight.shape[0]
model.config.name_or_path = 'dummy-foo/chatglm3-japanese-zero'
print(f'tokenizer size: {len(tokenizer)}, pad_to_multiple_of=64, model embedding size: {model.get_input_embeddings().weight.shape}')

# save resized tokenizer and resized model to one dir
need_to_save_model = True
if need_to_save_model:
    tokenizer.save_pretrained(CHATGLM_RESIZED_MODEL_DIR)
    model.save_pretrained(CHATGLM_RESIZED_MODEL_DIR)

    files = glob.glob(f'{CHATGLM_RESIZED_CODE_DIR}/*.py')
    for file in files:
        shutil.copy(file, CHATGLM_RESIZED_MODEL_DIR)

    print(f'model saved to {CHATGLM_RESIZED_MODEL_DIR}, now you can load model with transformers.AutoXXX')