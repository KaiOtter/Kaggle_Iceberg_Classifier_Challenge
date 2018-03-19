import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import train_test_split
import csv

"""
test for dealing this data
Use linear-regression on band_1 
"""

filepath = "train.json"
df = pd.read_json(filepath, typ="frame")
label = df["is_iceberg"]
train = df["band_1"]
y = []
for l in label:
    y.append(int(l))
x = []
for t in train:
    x.append(t)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred = linreg.predict(x_test)
correct = 0
for i in range(len(y_pred)):
    if y_pred[i] > 0.5 and y_test[i] == 1:
        correct += 1
    elif y_pred[i] <= 0.5 and y_test[i] == 0:
        correct += 1
print("acc : {}".format(correct/len(y_test)))

csvFile = open('submission_no_01.csv','w', newline='')
writer = csv.writer(csvFile)
head = ["id", "is_iceberg"]
writer.writerow(head)
test_df = pd.read_json("test.json", typ="frame")
test_band1 = test_df["band_1"]
test_id = test_df["id"]
test_x = []
for x in test_band1:
    test_x.append(x)
test_pred = linreg.predict(test_x)
for i in range(len(test_pred)):
    v = test_pred[i]
    if test_pred[i]<0:
        v = 0
    elif test_pred[i]>1:
        v = 1
    writer.writerow([test_id[i], v])
