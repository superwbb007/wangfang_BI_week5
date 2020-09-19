import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat,get_feature_names

#数据加载
data = pd.read_csv("movielens_sample.txt")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['rating']

# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
print(fixlen_feature_columns)
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)
#结果部分
"""Train on 128 samples, validate on 32 samples
2020-09-17 22:01:22.348549: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-17 22:01:22.362630: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fb810d5c7b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-17 22:01:22.362644: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
128/128 [==============================] - 0s 3ms/sample - loss: 14.3828 - mean_squared_error: 14.3828 - val_loss: 13.8707 - val_mean_squared_error: 13.8707
test RMSE 3.661379521437241"""