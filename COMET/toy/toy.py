from comet.models import download_model, load_checkpoint
import torch
# model = download_model("wmt-large-da-estimator-1719")
# estimator model checkpoint location:
route = "/home/steven/.cache/torch/unbabel_comet/wmt-large-da-estimator-1719/_ckpt_epoch_1.ckpt"
model = load_checkpoint(route)
# the encoder uses sentencebpe, following GPT-2 process, the model is xmlr.base's sentencepiece model
# temp = [
#     {
#         "src": "Dem Feuer konnte Einhalt geboten werden",
#         "mt": "The fire could be stopped",
#         "ref": "They were able to control the fire."
#     },
#     {
#         "src": "Schulen und Kindergärten wurden eröffnet.",
#         "mt": "Schools and kindergartens were open",
#         "ref": "Schools and kindergartens opened"
#     }
# ]
# model.predict(temp)
# print(model.encoder.tokenizer.encode("a e i de"))
# print(model)

# | dictionary: 250001 types
# | num. model params: 583740589 (num. trained: 0)
# Avg Score is:  0.14427588629862292 for zh-en trained for 4 epochs

# prepare model 
with open('gen-zh-en.out', 'r') as f:
    data = f.read()

source = []
hypothesis = []
reference = []

for line in data.split('\n'):
    # find source, hypothesis, and reference
    if line.startswith('S'):
        source.append(line.split('\t')[1])
    elif line.startswith('T'):
        reference.append(line.split('\t')[1])
    elif line.startswith('H'):
        hypothesis.append(line.split('\t')[2]) 
    else:
        continue

data = {"src": source, "mt": hypothesis, "ref": reference}
data = [dict(zip(data, t)) for t in zip(*data.values())]

prediction = model.predict(data, cuda=True, show_progress=True)
total_score = 0
size = 0
for idx, item in enumerate(prediction[0]):
    score = item['predicted_score']
    print("Score for {}th sentence is {}".format(idx, score))
    total_score += float(score)
    size += 1
avg_score = total_score / size
print('Avg Score is: ', avg_score)

# COMET
# 他表示:“作为一家电力公司,这对于我们来说可不是一个好消息”。
# 他说,“成为一个 电力的公司的、 这对我而言 可没有 一个 好信”。

# 他表示:“作为一家电力公司,这对于我们来说可不是一个好消息”。
# 我说,“成为一个 电力的公司的、 这 对于 我们而言 可并不是 一个好的信息”。

# BLEU
# 他表示:“作为一家电力公司,这对于我们来说可不是一个好消息”。
# 他称,“ 作为一个 电力的 公司 , 这对 我们而言 可不 一个密切通知”。

# 他表示:“作为一家电力公司,这对于我们来说可不是一个好消息”。
# 他表明“ 作为一个 电力的公司的。 这对 我们而言 可并非 一个很信息”。