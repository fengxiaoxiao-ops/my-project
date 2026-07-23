import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. 论文绘图环境配置
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="SimHei")

# 2. 构造模拟 C 题数据 (假设预测目标是: 销量)
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    '商品单价': np.random.uniform(10, 100, n_samples),
    '推广费用': np.random.uniform(1000, 5000, n_samples),
    '店铺评分': np.random.uniform(3.0, 5.0, n_samples),
    '物流距离': np.random.uniform(10, 800, n_samples)
})
# 设定内在机理：推广费用和店铺评分对销量起正向作用，单价起反向作用
data['历史销量'] = (data['推广费用'] * 0.5
                    + data['店铺评分'] * 200
                    - data['商品单价'] * 15
                    + np.random.normal(0, 200, n_samples))

# 3. 划分特征 (X) 和 目标变量 (y)
X = data.drop('历史销量', axis=1)
y = data['历史销量']

# 划分训练集 (80%) 和 测试集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构建并训练随机森林回归模型
# n_estimators: 树的数量, max_depth: 树的最大深度(防止过拟合)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)

# 5. 模型预测与评估
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== 模型预测性能评估 ===")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"拟合优度 (R^2): {r2:.4f} (越接近1说明预测越准)")

# 6. 提取并绘制“特征重要性”图表 (核心加分项)
importances = rf_model.feature_importances_
feature_names = X.columns
# 将特征重要性排序
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5), dpi=100)
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title('随机森林预测模型 - 各指标特征重要性分析', fontsize=10, pad=15)
plt.xlabel('对目标变量的贡献度 (0~1)', fontsize=10)
plt.ylabel('评价指标', fontsize=10)
plt.tight_layout()

# 导出高质量矢量图
plt.savefig('feature_importance.svg', format='svg')
plt.show()