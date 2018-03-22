import pandas as pd

filepath = "train.json"
df = pd.read_json(filepath, typ="frame")
tt = pd.DataFrame()
inc_angle = df["inc_angle"]
label = df["is_iceberg"]
s = "total samples : {}".format(len(inc_angle))
print(s)
account = 0
pos = 0
neg = 0
for i in range(len(inc_angle)):
    if type(inc_angle[i]) != type(1.523):
        account += 1
    if label[i] == 1:
        pos += 1
    elif label[i] == 0:
        neg += 1
s = "inc_angle has {} na".format(account)
print(s)
assert (pos + neg) == len(inc_angle) == len(label)
s = "positive : {}, negative : {}".format(pos, neg)
print(s)
