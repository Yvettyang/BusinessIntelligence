# 数据预处理
import pandas as pd

rawdata = pd.read_csv('./datasets_8127_11403_Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, rawdata.shape[0]):
    temp = []
    for j in range(0, rawdata.shape[1]):
        if str(rawdata.values[i,j]) != 'nan':
            temp.append(str(rawdata.values[i,j]))
    transactions.append(temp)

# 数据挖掘
from efficient_apriori import apriori

itemsets, rules = apriori(transactions, min_support = 0.03, min_confidence = 0.4)
print('频繁项集：', itemsets)
print('关联规则：', rules)
