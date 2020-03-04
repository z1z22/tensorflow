import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import logging
logging.basicConfig(filename="data/facebook.log",level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#1数据准备集处理
#1.1调取数据
data = pd.read_csv('data/FBlocation/train.csv')
data = data.query('x<2&y<2')
logging.info('选取的数据：')
logging.info(data)
logging.info(data.shape)
#1.2处理时间特征
time_value = pd.to_datetime(data['time'],unit='s')
date = pd.DatetimeIndex(time_value)
data['day'] = date.day
data.loc[:,'weekday'] = date.weekday
data['hour'] = date.hour
#1.3过滤掉签到较少的点
placecount = data.groupby('place_id').count()['row_id']
data_final = data[data['place_id'].isin(placecount[placecount >5].index.values)]
#1.4筛选特征值和目标值
x = data_final[['x','y','day','weekday','hour']]
y = data_final['place_id']
logging.info('数据准备完成')
logging.info('数据集')
logging.info(x)

logging.info('目标集')
logging.info(y)





#2机器学习
#2.1数据集划分
x_train,x_test,y_train,y_test = train_test_split(x,y)
logging.info('数据集划分完成')

#2.2特征工程, 标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)#用训练集的fit来对测试集进行标准化
logging.info('数据标准化完成')
logging.info(x_train)
# 2.3KNN预估器流程
estimator = KNeighborsClassifier()#n_neighbors即k值在后面调优，这里不设置

#加入网格搜索与k值调优
#准备参数
param_dict = {'n_neighbors':[5,7,9]}
estimator = GridSearchCV(estimator, param_grid=param_dict,cv=5)#3指3折调优
estimator.fit(x_train,y_train)
logging.info('训练完成，评估结果：')
# 3模型评估
#3.1方法1，直接比对
y_predict = estimator.predict(x_test)
logging.info('y_predict')
logging.info(y_predict)
logging.info('直接比对真实值和预测值')
logging.info(y_test == y_predict)
#3.2方法2，计算准确率
score = estimator.score(x_test,y_test)
logging.info('准确率为：')
logging.info(score)

#3.3查看调优结果
#最佳参数
logging.info('最佳参数')
logging.info(estimator.best_params_)
#最佳结果
logging.info('最佳结果:')
logging.info(estimator.best_score_)
#最佳估计器
logging.info('最佳估计器：')
logging.info(estimator.best_estimator_)
#交叉验证结果
logging.info('交叉验证结果:')
logging.info(estimator.cv_results_)