import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset = pd.read_csv('E:/cxs/机器学习/100days/datas/data1.csv')
print(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
# iloc 可以根据行列切分数据，逗号前后分别表示 行和列，单“：”表示所有， “1：3”表示从第2行（index 为1）到第第三行（indez为2）
# 这里 X Y 就把数据切分成了 前三列和 最后一列
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categories='auto', sparse=False)
X = onehotencoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("****")
#print(X)
print(X[:, :4])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_test)