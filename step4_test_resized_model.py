from constants import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_RESIZED_MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(CHATGLM_RESIZED_MODEL_DIR, trust_remote_code=True, device_map='cuda:0', torch_dtype=torch.float16)

# test Chinese chat, 预期得到和原生模型一样的答案
response, _ = model.chat(tokenizer, "你是谁", history=[])
print("\nQ：你是谁，A：", response)
response, _ = model.chat(tokenizer, "你喜欢甜粽子还是咸粽子", history=[])
print("\nQ：你喜欢甜粽子还是咸粽子，A：", response)

# test Japanese chat, まったく無意味な答えが得られるでしょう
response, _ = model.chat(tokenizer, "あなたは誰ですか", history=[])
print("\nQ：あなたは誰ですか，A：", response)
response, _ = model.chat(tokenizer, "すみません、ちょっとお聞きしたいことがあるんですが", history=[])
print("\nQ：すみません、ちょっとお聞きしたいことがあるんですが，A：", response)

# test Japanese tokenizer
print("\ntokens: ", tokenizer.tokenize("恥の多い生涯を送って来ました。自分には、人間の生活というものが、見当つかないのです。"))

