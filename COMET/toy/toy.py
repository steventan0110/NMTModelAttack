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

# 它使用 了10 000吨钢材、 20,000公吨混合凝土 和46根斜拉索。
# 国际奥委会和世界各国的媒从业看来 都 已经要迎 接这一轮挑战 了。
# 级医生与医科学生 7月份投票抵制 与 英国医学协会达成的协议事务
# 从面上看,网络社交平台 扩大了交间,降低了人们的唯一感。
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0108, -0.0042, -0.0021, -0.0054,  0.0124],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0053,  0.0022, -0.0030,  0.0007, -0.0012]], device='cuda:0',
#        grad_fn=<SliceBackward>)
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0098, -0.0052, -0.0031, -0.0064,  0.0134],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0043,  0.0012, -0.0040, -0.0003, -0.0002]], device='cuda:0',
#        grad_fn=<SliceBackward>)


# 它了10,000公吨钢材、20,000公混土根拉
# 国际奥会和各国的从业者,都已经要迎接这一轮挑战
# 初与医科学生于7投票抵制与英国学达成的合同交易。
# 从看网络台扩大了交际圈,了人们的孤感
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0108, -0.0042, -0.0021, -0.0054,  0.0124],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0053,  0.0022, -0.0030,  0.0007, -0.0012]], device='cuda:0',
#        grad_fn=<SliceBackward>)
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0098, -0.0052, -0.0031, -0.0064,  0.0134],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0063,  0.0012, -0.0040, -0.0003, -0.0022]], device='cuda:0',
#        grad_fn=<SliceBackward>)




# 它了10,000公吨钢材、20,000公混土根拉
# 国际奥会和各国的从业者,都已经要迎接这一轮挑战
# 初与医科学生于7投票抵制与英国学达成的合同交易。
# 从看网络台扩大了交际圈,了人们的孤感
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0108, -0.0042, -0.0021, -0.0054,  0.0124],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0053,  0.0022, -0.0030,  0.0007, -0.0012]], device='cuda:0',
#        grad_fn=<SliceBackward>)
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0098, -0.0032, -0.0031, -0.0064,  0.0114],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0063,  0.0012, -0.0040, -0.0003, -0.0022]], device='cuda:0',
#        grad_fn=<SliceBackward>)
# 这试图自杀的女性于周四清晨在速上辆辗目前警正在寻找击者
# 下半场刚开场费尔南多就以突破侧任意球,可小角度攻门打。
# ,2016年上半在西投资已达到338亿美元和去年同期几乎了10%。
# 尽管进出口值同比下降,但数据了,这自今年以来,连续五。
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0098, -0.0032, -0.0031, -0.0064,  0.0114],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0063,  0.0012, -0.0040, -0.0003, -0.0022]], device='cuda:0',
#        grad_fn=<SliceBackward>)
# tensor([[-0.0435,  0.0292, -0.0642,  0.0155,  0.0007],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0105, -0.0024, -0.0034, -0.0068,  0.0105],
#         [ 0.0103, -0.0385, -0.0219,  0.0487,  0.0447],
#         [ 0.0058,  0.0002, -0.0050, -0.0004, -0.0031]], device='cuda:0',
#        grad_fn=<SliceBackward>)