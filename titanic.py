from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

# 数据探索
# print(train_data.info()) # 数据信息：列名、非空个数、类型等
# print(train_data.describe()) # 数据摘要：数量、平均值、标准差、最值、分位数等
# print(train_data.describe(include=['O'])) # 离散数据分布
# print(train_data.head()) # 前5条数据
# print(train_data.tail()) # 后5条数据

# 缺失值处理
# 使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
# 使用票价的均值填充票价中的nan值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
# 使用登录最多的港口来填充登录港口的nan值
# print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
# print(dvec.feature_names_)
train_features = pd.DataFrame(train_features)
X_train, X_test, y_train, y_test = train_test_split(train_features.astype(np.float64),
    train_labels.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
# print(tpot.score(X_test, y_test))
tpot.export('tpot_titanic_pipeline.py')

submission = tpot.predict(test_features)
final = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': submission})
final.to_csv('./submission.csv', index = False)