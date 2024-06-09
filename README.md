# ChatGLM3-Japanese
[ChatGLM3-6B](https://github.com/THUDM/ChatGLM3)是一个中英双语大模型，本项目为ChatGLM3-6B加入日文能力。  

step1-4是扩词表和resize模型，step5-6是训练resize后的模型。

HuggingFace链接：
- [ChatGLM3-Japanese-Zero]( https://huggingface.co/dummy-foo/ChatGLM3-Japanese-Zero )：经过扩词表和resize后的模型，保留了ChatGLM3的中英文能力，尚无日文能力，但因为编码效率高，适合在日文语料上训练。
- [ChatGLM3-Japanese](https://huggingface.co/dummy-foo/ChatGLM3-Japanese)：对ChatGLM3-Japanese-Zero进行日文语料增量预训练和指令微调的模型，可以日文对话。


## 安装依赖
若只需要运行step1-4，即只需要训练tokenizer，则如下安装：
```
pip install -r requirements.txt
```
若需要运行step5-6，即增量预训练和指令微调，因本仓库使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)框架，本质就是要安装它的依赖，方便起见整理在了requirements_llama_factory.txt中，如下安装：
``` 
pip install -r requirements_llama_factory.txt
```

## 运行代码
### step1 训练日文tokenizer
使用1GB日文语料（800MB日文wiki + 200MB青空文库）训练日文tokenizer，使用sentencepiece包提供的BPE算法，词表20000.
```
python step1_train_tokenizer.py
```

ChatGLM原生tokenizer对日文的分词效果，基本每个汉字和假名都被分为1个token，ぬ甚至被分为3个tokens，编码效率很低。

```
tokenizer.tokenize('拝借させてくださいませんか')
['▁', '拝', '借', 'さ', 'せ', 'て', 'く', 'だ', 'さ', 'い', 'ま', 'せ', 'ん', 'か']
tokenizer.tokenize('読んでもよろしいでしょうか')
['▁', '読', 'ん', 'で', 'も', 'よ', 'ろ', 'し', 'い', 'で', 'し', 'ょ', 'う', 'か']
```
训练得到的tokenizer对日文的分词效果，基本都切分为了有意义的块，编码效率很高。
（吐槽一下，虽然させて、ください、ません、か每个词都很有意义，但它们组合成的させてくださいませんか就没什么意义。。。）
```
print(sp.encode_as_pieces('拝借させてくださいませんか'))
['▁', '拝', '借', 'させて', 'ください', 'ません', 'か']
print(sp.encode_as_pieces('読んでもよろしいでしょうか'))
['▁', '読', 'んでも', 'よろしい', 'でしょうか']
```

### step2 合并ChatGLM原生tokenizer和日文tokenizer
合并后词表大小64789 -> 78554
```
python step2_merge_tokenizer.py
```
期望不影响对中文的分词，测试一下，ChatGLM原生和本项目的tokenizer对中文分词完全一致，符合预期。  
``` 
text = "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"

pre tokenizer: ['▁白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上一', '层楼', '。']

cur tokenizer: ['▁白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上一', '层楼', '。']
```


再来对比一下对日文的分词，原文长度127个字符，原生和本项目分别编码为125和69个tokens，每个字符的token数从1降到了0.54，编码效率几乎翻倍。
``` 
text = """どうしてこうなるんだろう…初めて、好きな人が出来た。一生ものの友だちができた。嬉しいことが二つ重なって、
その二つの嬉しさが、また、たくさんの嬉しさを連れてきてくれて。夢のように幸せな時間を手に入れたはずなのに…なのに、どうして、こうなっちゃうんだろう…"""

pre tokenizer, len=125: ['▁', 'ど', 'う', 'し', 'て', 'こ', 'う', 'な', 'る', 'ん', 'だ', 'ろ', 'う', '…', '初', 'め', 'て', '、', '好', 'き', 'な', '人', 'が', '出来', 'た', '。', '一生', 'も', 'の', 'の', '友', 'だ', 'ち', 'が', 'で', 'き', 'た', '。', '嬉', 'し', 'い', 'こ', 'と', 'が', '二', 'つ', '重', 'な', 'っ', 'て', '、', '<0x0A>', 'そ', 'の', '二', 'つ', 'の', '嬉', 'し', 'さ', 'が', '、', 'ま', 'た', '、', 'た', 'く', 'さ', 'ん', 'の', '嬉', 'し', 'さ', 'を', '連', 'れ', 'て', 'き', 'て', 'く', 'れ', 'て', '。', '夢', 'の', 'よ', 'う', 'に', '幸', 'せ', 'な', '時間', 'を', '手', 'に', '入', 'れ', 'た', 'は', 'ず', 'な', 'の', 'に', '…', 'な', 'の', 'に', '、', 'ど', 'う', 'し', 'て', '、', 'こ', 'う', 'な', 'っ', 'ち', 'ゃ', 'う', 'ん', 'だ', 'ろ', 'う', '…']

cur tokenizer, len=69: ['▁', 'どうして', 'こう', 'なる', 'んだろう', '…', '初めて', '、', '好きな', '人が', '出来た', '。', '一生', 'ものの', '友', 'だち', 'が でき', 'た', '。', '嬉', 'しい', 'ことが', '二つ', '重な', 'って', '、', '<0x0A>', 'その', '二つの', '嬉', 'しさ', 'が', '、', 'また', '、', 'たくさん', 'の', '嬉', 'しさ', 'を連れて', 'きて', 'くれ', 'て', '。', '夢', 'のよ', 'う', 'に', '幸', 'せ', 'な', '時間を', '手に', '入れた', 'はず', 'なの', 'に', '…', 'なの', 'に', '、', 'どうして', '、', 'こう', 'な', 'っちゃ', 'うん', 'だろう', '…']
```

### step3 resize模型的输入输出embedding后导出模型
比较容易出错的一步，在blog中有详细介绍。运行完后就可以愉快地使用transformers.AutoXXX来加载了。
``` 
python step3_create_resized_model.py
```

### step4 测试ChatGLM3-Japanese-Zero模型
测试扩了词表后模型的效果，预期中文能力无变化（原生tokenizer能力都被保留了），日文能力完全丧失（因为新扩的token的embedding是随机初始化的）
``` 
python step4_test_resized_model.py
```
代码内容：
``` 
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
```
看一下结果，符合预期。中文很完美，日文完全不会。下面就开始训练了，因为编码效率翻倍，训练效率也会高很多。
``` 
Q：你是谁，A： 我是一个名为 ChatGLM3-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户的问题和要 求提供适当的答复和支持。

Q：你喜欢甜粽子还是咸粽子，A： 作为一个人工智能助手，我没有口味和个人喜好。但我可以帮助您了解甜粽子和咸粽子的一些特点。

甜粽子通常以糯米、红枣、莲蓉等甜馅为主，口感甜腻，适合喜欢甜食的人。
咸粽子则以糯米、肉、蛋黄等咸馅为主，口感咸鲜，适合喜欢重口味的人。

您可以根据自己的口味和喜好尝试一下甜粽子和咸粽子，看看哪种更适合您。

Q：あなたは誰ですか，A： 抱歉，我不太明白您的问题。能否请您提供更多上下文或信息，以便我更好地理解您的问题？

Q：すみません、ちょっとお聞きしたいことがあるんですが，A： 这句话看起来像是日语和英语混杂的语言，其中“おいしました”的意思是“很好吃”，“红色的”则明确指出了事物的 颜色。因此，这句话的意思可能是“这很红，很好吃”，但这种表达方式并不常见。如果你想表达“这很红，很好吃”，可以用英语的“This is very red and delicious”来表达。

tokens: ['▁', '恥', 'の多い', '生涯', 'を送', 'って', '来ました', '。', '自分に', 'は', '、', '人間の', '生活', 'というもの', 'が', '、', '見', '当', 'つかない', 'のです', '。']
```


### step5 增量预训练
使用1B tokens进行增量预训练，日文、中文、英文占比分别约76%、12%、12%，deepseed-zero2，四卡4090约60小时。

单卡训练（供单步调试）
``` 
python step5_train_pt_single_gpu.py
```
多卡训练（供真实运行）
```
sh step5_train_pt_multi_gpu.sh
```

### step6 指令微调
使用22万条指令数据集进行指令微调，日文、中文、英文、代码占比分别约52%、22%、9%、17%，deepseed-zero2，四卡4090约4.5小时  

单卡训练（供单步调试）
``` 
python step6_train_sft_single_gpu.py
```
多卡训练（供真实运行）
```
sh step6_train_sft_multi_gpu.sh
```