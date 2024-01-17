import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
import time

# 读取 CSV 文件
data = pd.read_csv('industrial_dataset.csv')

# 获取特征和标签列
X = data.iloc[:, 1:-1]  # 获取所有特征列
y = data.iloc[:, -1]   # 获取最后一列作为标签列

# 初始化 SVM 模型
model = SVR()

# 初始化 K 折交叉验证
k_fold = KFold(n_splits=8, shuffle=True, random_state=42)  # 这里选择将数据集分成 8 折

# 使用交叉验证评估模型性能
mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=k_fold)

# 将负的均方误差转换为正数
mse_scores = -mse_scores

# 创建一个列表来存储每个折叠的 R² 值
r_squared_scores = []
start_time = time.time()
# 打印每折交叉验证的 MSE 和 R² 分数
for i, mse_score in enumerate(mse_scores):
    print(f"Fold {i+1} MSE: {mse_score}")
    
    # 拟合模型并进行预测
    train_index, test_index = list(k_fold.split(X))[i]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 计算 R²
    r_squared = r2_score(y_test, y_pred)
    r_squared_scores.append(r_squared)
    
    print(f"Fold {i+1} R²: {r_squared}")

# 打印平均 MSE 和 R²
print(f"\nAverage MSE: {mse_scores.mean()}")
print(f"Average R²: {sum(r_squared_scores) / len(r_squared_scores)}")
end_time = time.time()
execution_time = end_time - start_time
print(f"代码执行时间为: {execution_time} 秒")
