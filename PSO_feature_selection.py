import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from pyswarm import pso
from sklearn.metrics import mean_squared_error, r2_score
import time

# 假设你有一个名为 'data' 的 Pandas DataFrame，其中包含特征和标签列
# 特征列为 'features'，标签列为 'label'
data = pd.read_csv('industrial_dataset.csv')
# 从 'data' 中分离特征和标签
data = data.iloc[:,1:]
all_feature = data.iloc[:, :-1]
all_label = data.iloc[:, -1]

X_scaled = all_feature
y = all_label

X_scaled = all_feature.values
y = all_label.values


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义适应度函数（目标函数），这里以线性回归的均方误差作为评价指标
def fitness_function(selected_features):
    selected_features = np.array(selected_features)
    selected_indices = np.where(selected_features == 1)[0]
    if len(selected_indices) == 0:
        return 999999  # 返回一个较大的值，表示无效的特征组合
    else:
        model = LinearRegression()        
        model.fit(X_train[:, selected_indices], y_train)
        y_pred = model.predict(X_test[:, selected_indices])
        mse = np.mean((y_test - y_pred) ** 2)
        # print(mse)
        return mse

# 定义 PSO 算法进行特征选择
def feature_selection_pso(fitness_function, num_features, num_particles=40, max_iter=100):
    lb = [0] * num_features  # 特征是否被选择的下界
    ub = [1] * num_features  # 特征是否被选择的上界

    def objective_function(selected_features):
        return fitness_function(selected_features)

    # 使用 PSO 算法优化特征选择
    best_features, _ = pso(objective_function, lb, ub, maxiter=max_iter, swarmsize=num_particles)
    return best_features

num_features = X_scaled.shape[1]  # 获取特征数量
selected_features = feature_selection_pso(fitness_function, num_features)
selected_indices = np.where(selected_features == 1)[0]
selected_feature_names = data.columns[selected_indices]

print("选择的特征列：", selected_feature_names)



# 定义线性回归模型训练与评估函数
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    return mse, r_squared

# K折交叉验证
k_fold = KFold(n_splits=8, shuffle=True, random_state=42)

# 存储每折交叉验证的MSE和R²
mse_scores = []
r_squared_scores = []
start_time = time.time()
for train_index, test_index in k_fold.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 使用选定的特征进行训练和评估
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    mse, r_squared = train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test)
    mse_scores.append(mse)
    r_squared_scores.append(r_squared)

# 计算平均MSE和R²
average_mse = np.mean(mse_scores)
average_r_squared = np.mean(r_squared_scores)
end_time = time.time()
execution_time = end_time - start_time
print("平均MSE:", average_mse)
print("平均R²:", average_r_squared)
print(f"代码执行时间为: {execution_time} 秒")

