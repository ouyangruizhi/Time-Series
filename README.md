# Time-Series📈
北京林业大学统计学zx老师【时间序列分析】课程作业与论文源码  
🚀🚀🚀**高分好课** 
***
### ARMA平稳序列建模
- 第三次作业（前两次是手写）
- 对应文件：ts3_220403102.py
- ✅任务：
  - 图检验判断平稳性
  - LB检验判断纯随机性
  - 利用相关图定阶
  - 参数估计与检验
  - 残差白噪声检验
  - 基于最佳准则函数选择模型
  - 区间预测
### ARIMA不平稳序列建模（趋势）
- 第四次作业
- 对应文件：ts4_220403102.ipynb
- ✅任务：
  - ADF平稳性检验
  - 差分
### SARIMA不平稳序列建模（趋势+季节）
- 第五次作业
- 对应文件：ts5_220403102.ipynb
- ✅任务：
  - 季节差分
### 指数平滑、回归+残差AR
- 第六次作业
- 对应文件：ts6_220403102.ipynb
- ✅任务：
  - 对数变换
  - 指数平滑（二次、三次）
  - 多项式拟合
  - 残差AR建模
### ARIMA建模流程整理
- 对应文件：ARIMA.py
- 🔥建模流程：图检验 -> 差分 -> ADF检验 -> 纯随机检验 -> 模型识别 -> 残差诊断 -> 区间预测
### ARIMA-ARCH
- 对应文件：ts6.5_220403102.ipynb
- ✅任务：
  - 平稳性：PP检验
  - 自相关检验
  - 异方差检验与残差ARCH建模
  - 残差的整体效果检验：白噪声检验与正态性检验
### 🔥ARIMA-AR-GARCH建模
- 第七次作业
- 对应文件：ts7_220403102.ipynb
- ✅任务：
  - 数据变换
  - 残差诊断与解决：ARIMA-AR-GARCH模型

采用**串联**的形式：
$$ARIMA(0,1,0)-AR(4)-GARCH(1,1)$$  
我的例子具体而言是：

$$\begin{cases}
X_t=X_{t-1}+\epsilon_t \\
\epsilon_t=0.0323\epsilon_{t-3}+0.0006\epsilon_{t-4}+v_t \\
v_t = \sqrt{h_t}e_t  \\ 
h_t = 1.9125e-07 + 0.0500h_{t-1} + 0.9300v_{t-1}^2
\end{cases}$$
### SETAR自激励门限自回归
- 第八次作业
- 对应文件：ts8_220403102.ipynb
- ✅任务：
  - SETAR建模
### HP滤波+傅立叶变换
- 第九次作业
- 对应文件：ts9_220403102.ipynb
- ✅任务：
  - hp滤波分解数据
  - 趋势采用holt指数平滑
  - 周期采用快速傅立叶变换谱分析
### VARMAX
- 第十次作业
- 对应文件：ts10_220403102.ipynb
- ✅任务：
  - 相关性分析
  - 格兰杰因果检验
  - VARMAX建模
### RNN循环神经网络
- 第十一次作业
- 对应文件：ts11_220403102.ipynb
- ✅任务：
  - CNN-BiLSTM-Attention
  - CNN-BiGRU-Attention
### 课程论文
- 《基于时间序列模型的广州碳交易权K线价格预测》
- 对应文件：tsa1_220403102.ipynb与tsa2_220403102.ipynb
- ✅任务：
  - ARIMA
  - 残差AR
  - 残差EGARCH
  - VARMA
  - 神经网络
***
__主要依赖库：[statsmodels](https://www.statsmodels.org/stable/index.html)、[arch](http://bashtage.github.io/arch/)、[pmdarima](https://pypi.org/project/pmdarima/)、[sklearn](https://scikit-learn.org/stable/)、[keras](https://keras-cn.readthedocs.io/en/latest/)、[tensorflow](https://tensorflow.google.cn/?hl=zh-cn)、[pytorch](https://pytorch-cn.readthedocs.io/zh/latest/)、[scipy](https://docs.scipy.org/doc/scipy-1.13.0/index.html)__   
__主要数据来源：[akshare](https://akshare.akfamily.xyz/)、[yfinance](https://pypi.org/project/yfinance/)、[Tushare](https://tushare.pro/document/2)、[歪枣网](http://waizaowang.com/api/detail/101)__  
如有任何问题，可以写[Issues](https://github.com/ouyangruizhi/Time-Series/issues)或者Email我：ouyangruizhi@bjfu.edu.cn📮  
如果对你有帮助可以给我一个Star🌟
