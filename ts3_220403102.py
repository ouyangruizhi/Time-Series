import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA

# 数据导入
data = [126.4, 82.4, 78.1, 51.1, 90.9, 76.2, 104.5, 87.4, 110.5, 25, 69.3, 53.5, 39.8, 63.6, 46.7, 72.9, 79.6, 83.6, 80.7, 60.3, 79, 74.4, 49.6, 54.7, 71.8, 49.1, 103.9, 51.6, 82.4, 83.6, 77.8, 79.3, 89.6, 85.5, 58, 120.7, 110.5, 65.4, 39.9, 40.1, 88.7, 71.4, 83, 55.9, 89.9, 84.8, 105.2, 113.7, 124.7, 114.5, 115.6, 102.4, 101.4, 89.8, 71.5, 70.9, 98.3, 55.5, 66.1, 78.4, 120.5, 97, 110]

## 问题(1)利用图检验判断平稳性，LB统计量判断纯随机性
# 计算均值和方差
mean = np.mean(data)
variance = np.var(data)

print(f'均值：{mean}，方差：{variance}')

# 绘制序列图做平稳性检验
plt.figure(figsize=(16, 9), dpi=150)
plt.plot(data, linestyle='-', color='b', marker='o', markerfacecolor='b')

# 绘制均值线
plt.axhline(y=mean, color='k', linestyle='--')
plt.xlabel('', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(fname='序列图.png')

# 纯随机检验
LB = acorr_ljungbox(data, lags=range(1, 21), boxpierce=True, return_df=True)
print(LB)

## 问题(2)通过相关图选择模型拟合序列
# 绘制自相关图和偏自相关图并识别模型
fig = plt.figure(figsize=(12, 4), dpi=150)
ax1 = fig.add_subplot(121)
sm.graphics.tsa.plot_acf(data, lags=25, ax=ax1)
ax2 = fig.add_subplot(122)
sm.graphics.tsa.plot_pacf(data, lags=25, ax=ax2)
plt.savefig(fname='相关图.png')

# 模型识别为MA(2)

# 参数估计
# 定义 MA(2)模型的对数似然函数
def ma2_loglikelihood(params, data):
    mu, theta1, theta2, sigma2 = params
    n = len(data)
    loglik = 0
    eps = [0] * 2
    for t in range(n):
        xt = data[t]
        et = xt - mu - theta1 * eps[0] - theta2 * eps[1]
        loglik -= 0.5 * (np.log(2 * np.pi * sigma2) + et**2 / sigma2)
        eps[1] = eps[0]
        eps[0] = et
    return loglik

# 矩估计得到的初始值
theta0 = [mean, 0.1, 0.1, variance]

# 使用最小化函数进行极大似然估计
result = minimize(lambda params: -ma2_loglikelihood(params, data), theta0, method='BFGS')

# 输出参数估计结果
mu_hat, theta1_hat, theta2_hat, sigma2_hat = result.x
print(f'均值估计值：{mu_hat}')
print(f'theta1估计值:{theta1_hat}')
print(f'theta2估计值:{theta2_hat}')
print(f'方差估计值：{sigma2_hat}')

# 模型检验
# 模型整体效果检验
epsilon_prev1 = 0
epsilon_prev2 = 0

predictions = []
residuals = []

for xt in data:
    # 计算误差项
    epsilon_t = xt - mu_hat - theta1_hat * epsilon_prev1 - theta2_hat * epsilon_prev2
    # 计算预测值
    xt_pred = mu_hat + theta1_hat * epsilon_prev1 + theta2_hat * epsilon_prev2
    predictions.append(xt_pred)
    residuals.append(xt - xt_pred)
    # 更新误差项
    epsilon_prev2 = epsilon_prev1
    epsilon_prev1 = epsilon_t

print("残差序列：", residuals)

# 检验残差序列为白噪声
LB = acorr_ljungbox(residuals, lags=range(1, 21), boxpierce=True, return_df=True)
print(LB)

# 模型的参数检验
# 计算标准误差
n = len(data)
se_mu = np.sqrt(sigma2_hat / n)
residuals_array = np.array(residuals)
se_theta1 = np.sqrt(sigma2_hat / np.sum(residuals_array**2))
se_theta2 = np.sqrt(sigma2_hat / np.sum(residuals_array**2))

# 计算 t 值
t_mu = mu_hat / se_mu
t_theta1 = theta1_hat / se_theta1
t_theta2 = theta2_hat / se_theta2

# 假设显著性水平为 0.05，自由度为 n - 2
critical_value = stats.t.ppf(1 - 0.05 / 2, n - 2)

print(f"mu 的 t 值为：{t_mu}，临界值为：{critical_value}")
print(f"theta1 的 t 值为：{t_theta1}，临界值为：{critical_value}")
print(f"theta2 的 t 值为：{t_theta2}，临界值为：{critical_value}")

## 模型的优化
# 尝试MA(2)模型
ma2_model = ARIMA(data, order=(0, 0, 2))
ma2_results = ma2_model.fit()
ma2_aic = ma2_results.aic

# 尝试AR(1)模型
ar1_model = ARIMA(data, order=(1, 0, 0))
ar1_results = ar1_model.fit()
ar1_aic = ar1_results.aic

# 尝试ARMA(1,1)模型
arma11_model = ARIMA(data, order=(1, 0, 1))
arma11_results = arma11_model.fit()
arma11_aic = arma11_results.aic

# 最佳准则函数选择模型
print(f'MA(2)模型的 AIC 值：{ma2_aic}')
print(f'AR(1)模型的 AIC 值：{ar1_aic}')
print(f'ARMA(1,1)模型的 AIC 值：{arma11_aic}')

## 问题(3)预测未来5个数据
# 预测未来 5 个时间点
num_steps = 5
predictions = ma2_results.forecast(steps=num_steps)
print(f'未来五年降雪量(单位：mm)：{predictions}')
# 绘制原始数据和预测值
plt.figure(figsize=(16, 9), dpi=150)
plt.plot(data, linestyle='-', color='b', marker='o', markerfacecolor='b', label='Original Data')
plt.plot(range(len(data), len(data) + num_steps), predictions, linestyle='--', color='r', marker='o', markerfacecolor='r', label='Predictions')
plt.plot([len(data)-1, len(data)], [data[-1], predictions[0]], color='r', linestyle='--')
plt.xlabel('Time', fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig(fname='预测结果.png')

'''应当改成区间预测'''