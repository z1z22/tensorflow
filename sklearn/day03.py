from sklearn. datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report,roc_auc_score
import pandas as pd
import numpy as np
def line_demo():
    '''线性回归：正规方程优化方法预测波士顿房价'''
    #1获取数据
    boston = load_boston()
    #2数据集划分
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    #3无量刚化：标准化数据集
    transfor = StandardScaler()
    x_train = transfor.fit_transform(x_train)
    x_test = transfor.transform(x_test)
    #4预估器
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)
    #5打印模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-偏置为：\n", estimator.intercept_)
    #5模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("正规方程-均方误差为：\n", error)


def sgdr_demo():
    '''线性回归：随机梯度下降优化方法预测波士顿房价'''
    #1获取数据
    boston = load_boston()
    #2数据集划分
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    #3无量刚化：标准化数据集
    transfor = StandardScaler()
    x_train = transfor.fit_transform(x_train)
    x_test = transfor.transform(x_test)
    #4预估器
    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=100, penalty="l2")#有许多可调参数用来优化。经验表明，SGD 在处理约 10^6 训练样本后基本收敛。因此，对于迭代次数第一个合理的猜想是 n_iter = np.ceil(10**6 / n)，其中 n 是训练集的大小。
    estimator.fit(x_train,y_train)
    #5打印模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-偏置为：\n", estimator.intercept_)
    print("实际迭代次数：\n", estimator.n_iter_)# 要达到停止条件的实际迭代次数。
    print("实际更新数：\n", estimator.t_)  # 训练期间执行的重量更新数。等于(n_iter_ * n_samples)

    #5模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差为：\n", error)


def ridge_demo():
    '''线性回归：岭回归优化方法预测波士顿房价'''
    #1获取数据
    boston = load_boston()
    #2数据集划分
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    print(x_train.shape)
    #3无量刚化：标准化数据集
    transfor = StandardScaler()
    x_train = transfor.fit_transform(x_train)
    x_test = transfor.transform(x_test)
    #4预估器
    estimator = Ridge(alpha=0.5, max_iter=3000)#有许多可调参数用来优化
    estimator.fit(x_train,y_train)
    #alpha 参数控制估计系数的稀疏度。


    # **保存模型
    # joblib.dump(estimator, "data/ridge_demo.pkl")
    #
    # estimator = joblib.load('data/ridge_demo.pkl')

    #5打印模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-偏置为：\n", estimator.intercept_)
    #5模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("岭回归-均方误差为：\n", error)
  
def ridge_demo2():
    '''岭回归加入设置正则化参数：广义交叉验证'''
    boston = load_boston()
    #2数据集划分
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)
    # #3无量刚化：标准化数据集
    # transfor = StandardScaler()
    # x_train = transfor.fit_transform(x_train)
    # x_test = transfor.transform(x_test)
    #4预估器
    estimator = RidgeCV(alphas=[0.1,1,10],cv=10)#RidgeCV 增加交叉验证
    estimator.fit(x_train,y_train)

    # **保存模型
    # joblib.dump(estimator, "data/ridge_demo.pkl")
    #
    # estimator = joblib.load('data/ridge_demo.pkl')

    #5打印模型
    print("岭回归CV-权重系数为：\n", estimator.coef_)
    print("岭回归CV-偏置为：\n", estimator.intercept_)
    print('alphe:\n',estimator.alpha_)
    #5模型评估
    y_predict = estimator.predict(x_test)
    # print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("岭回归CV-均方误差为：\n", error)

def cancer_logi(): 
    '''逻辑回归预测癌症良恶性,属于分类算法'''
    #1、获取数据
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli', 'Mitoses', 'Class']
    cancer = pd.read_csv(path, names=column_name)
    #1处理数据
    #1.1.1 替换缺失值为np.nan
    cancer.replace(to_replace ='?',value=np.nan,inplace=True)
    #1.2.2 对数据进行清洗
    cancer.dropna(inplace=True)
    #1.2划分数据集与目标值
    x = cancer.iloc[:,1:-1]
    y = cancer.iloc[:,-1]
    # print(x,y)

    #2划分数据集
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=23)
    #3特征工程：标准化
    transtor = StandardScaler()
    x_train = transtor.fit_transform(x_train)
    x_test = transtor.transform(x_test)
    #4预估器
    estimator = LogisticRegression()
    estimator.fit(x_train,y_train)
    #打印预估器模型参数：
    print('回归系数：\n',estimator.coef_)#回归系数
    print('偏执:\n',estimator.intercept_)#偏执    

    #5.1对预估器进行保存
    # joblib.dump(estimator,'data/cancer_ligi.pkl')
    #5.2对预估器进行读取
    # estimator = joblib.load('data/cancer_ligi.pkl')
    #6模型评估
    #6.1直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print(y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    # 6.2计算准确率
    score = estimator.score(x_test,y_test)
    print('准确率评分为：',score)
    #6.3 查看精确率、召回率、F1-score
    report = classification_report(y_test,y_predict,labels=[2,4],target_names=['良性','恶性'])
    print('精确率、召回率、F1-score报告：\n',report)
    #6.4roc_auc_score
    #6.4.1转化结果为0，1，用于二分类评估
    # y_true = np.where(y_test>2,1,0)
    # y_score = np.where(y_predict)
    ra_score = roc_auc_score(y_test,y_score=y_predict)
    print('roc_auc_score：',ra_score)




if __name__ == "__main__":
    # line_demo()
    sgdr_demo()
    # ridge_demo()
    # ridge_demo2()
    # cancer_logi()
