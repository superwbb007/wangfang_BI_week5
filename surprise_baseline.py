#from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold
#import pandas as pd

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# ALS优化 
#bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
# SGD优化
bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
#algo = BaselineOnly()
#algo = NormalPredictor()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
    #计算MAE
    accuracy.mae(predictions,verbose=True)

uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
#以下为 ALS优化 结果部分
"""Estimating biases using als...
RMSE: 0.8627
Estimating biases using als...
RMSE: 0.8630
Estimating biases using als...
RMSE: 0.8660
user: 196        item: 302        r_ui = 4.00   est = 4.03   {'was_impossible': False}"""
#以下为 SGD 优化 结果部分
"""Estimating biases using sgd...
RMSE: 0.8748
Estimating biases using sgd...
RMSE: 0.8733
Estimating biases using sgd...
RMSE: 0.8749
user: 196        item: 302        r_ui = 4.00   est = 3.88   {'was_impossible': False}
"""
"""zhangyuxi@ZhangdeMacBook-Pro MovieLens % python3 surprise_baseline.py
Estimating biases using sgd...
RMSE: 0.8744
MAE:  0.6738
Estimating biases using sgd...
RMSE: 0.8746
MAE:  0.6741
Estimating biases using sgd...
RMSE: 0.8742
MAE:  0.6742
user: 196        item: 302        r_ui = 4.00   est = 4.06   {'was_impossible': False}"""
#Baseline算法是基于统计的基准预测线打分,通过设立基线，并引入用户的偏差以及item的偏差
