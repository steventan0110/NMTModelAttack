from comet.models import download_model, load_checkpoint

# model = download_model("wmt-large-da-estimator-1719")
# estimator model checkpoint location:
route = "/home/steven/.cache/torch/unbabel_comet/wmt-large-da-estimator-1719/_ckpt_epoch_1.ckpt"
model = load_checkpoint(route)
print(model)
# print(model)

# prepare model 
with open('gen.out', 'r') as f:
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