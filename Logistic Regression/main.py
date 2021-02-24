from LogisticRegression import LogisticRegression
import pandas as pd
import numpy as np

# data = pd.read_csv('bank-additional/')

model = LogisticRegression()

train_x = [1.2,2.4,4.8,2.1,9.3,8.7,7.9,10.1]
train_y = [0,0,0,0,1,1,1,1]

model.fit(train_x,train_y)

for x in train_x:
    print(model.predict(x))
