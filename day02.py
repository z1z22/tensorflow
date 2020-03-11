from sklearn.datasets import load_iris
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV#网格搜索与交叉验证
from sklearn.feature_extraction import DictVectorizer#字典特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer#tf_idf特征抽取
from sklearn.preprocessing import StandardScaler#无量刚化：标准化
from sklearn.neighbors import KNeighborsClassifier#K近邻算法
from sklearn.tree import DecisionTreeClassifier,export_graphviz#决策树及可视化
from sklearn.naive_bayes import MultinomialNB#朴素贝叶斯算法
from sklearn.ensemble import RandomForestClassifier#随机森林算法
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import classification_report,roc_auc_score

import pandas as pd
# import logging
# logging.basicConfig(filename="data/day2_1.log",level=logging.INFO,
                    # format='%(asctime)s - %(levelname)s - %(message)s')

def knn_iris():
    '''案例1：K近邻对鸢尾花种类预测'''
    # 1）获取数据
    iris = load_iris()
    # 2）数据集划分
    x_train,x_test,y_train, y_test= train_test_split(iris.data, iris.target, random_state =22)#随机数种子random_state会影像预测结果

    # 3）特征工程, 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)#用训练集的fit来对测试集进行标准化
    # 4）KNN预估器流程
    estimator = KNeighborsClassifier(n_neighbors=3)#n_neighbors即k值选择不当结果不准确
    estimator.fit(x_train,y_train)
    # 5）模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('knn_iris准确率为：\n',score)

def knn_iris_gscv():
    '''案例1：鸢尾花种类预测,增加k值调优'''
    # 1）获取数据
    iris = load_iris()
    # 2）数据集划分
    x_train,x_test,y_train, y_test= train_test_split(iris.data, iris.target, random_state =6)#random_state 随机数种子会影像预测结果

    # 3）特征工程, 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)#用训练集的fit来对测试集进行标准化
    # 4）KNN预估器流程
    estimator = KNeighborsClassifier()#n_neighbors即k值在后面调优，这里不设置

    #加入网格搜索与k值调优
    #z准备参数
    param_dict = {'n_neighbors':[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict,cv=10)#10指10折调优
    
    estimator.fit(x_train,y_train)
    # 5）模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('准确率为：\n',score)

    #查看调优结果
    #最佳参数
    print('最佳参数：\n',estimator.best_params_)
    #最佳结果
    print('最佳结果:\n',estimator.best_score_)
    #最佳估计器
    print('最佳估计器：\n',estimator.best_estimator_)
    #交叉验证结果
    print('交叉验证结果:\n',estimator.cv_results_)

def nb_demo():
    '''用朴素贝叶斯算法对新闻分类'''
    #1、获取数据
    news = fetch_20newsgroups(subset ='all')
    # print(news.data[1])
    # print(news)


    # 2、划分数据集
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target)
    
    #3、特征工程，文本特征读取-tfidf
    transfor = TfidfVectorizer(min_df=4, #严格忽略低于给出阈值的文档频率的词条，语料指定的停用词。
                            # analyzer='word', #定义特征为词（word）
                            # ngram_range=(1, 3), #ngram_range(min,max)是指将text分成min，min+1，min+2,.........max 个不同的词组
                            # use_idf=1, #使用idf重新计算权重
                            # smooth_idf=1, #分母加一
                            # sublinear_tf=1, #线性缩放TF
                            stop_words='english' #忽略英文停用词
                            )
    x_train = transfor.fit_transform(x_train)
    x_test = transfor.transform(x_test)
    print(x_train[1])
    print(x_train.shape)
    #4、朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    #5、模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('准确率为：\n',score)

def tree_iris():
    '''用决策树对鸢尾花种类预测'''
    # 1）获取数据
    iris = load_iris()
    # 2）数据集划分
    x_train,x_test,y_train, y_test= train_test_split(iris.data, iris.target, random_state =22)#random_state会影像预测结果

    # 3）特征工程, 决策树可以不进行标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)#用训练集的fit来对测试集进行标准化
    # 4）决策树预估器流程
    estimator = DecisionTreeClassifier(criterion='entropy',max_depth=None)
    estimator.fit(x_train,y_train)
    # 5）模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('tree_iris准确率为：\n',score)

    #对决策树进行可视化
    export_graphviz(estimator,out_file='data/iris_tree.dot', max_depth=None, feature_names=iris.feature_names)

def titanic_tree():
    '''决策树计算Titanic获救'''
    #1 获取泰坦尼克获救人员数据
    df = pd.read_csv('data/titanic.csv')
    #2处理数据
    #2.1处理age缺失值
    df['age'].fillna(df['age'].mean(),inplace=True)
    x = df[['pclass','age','sex']]
    y = df['survived']
    #2.2转化为字典，由于age，pclass都是分类特征，最好转化成字典，然后用字典特征抽取
    x = x.to_dict(orient="records")
    #3数据集划分
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=6)
    #4字典特征抽取
    transtor = DictVectorizer()
    x_train = transtor.fit_transform(x_train)
    x_test = transtor.transform(x_test)
    #5决策树预估器流程
    estimator= DecisionTreeClassifier(
        # criterion='entropy', 
        max_depth=6)
    estimator.fit(x_train,y_train)
    #6模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    # print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('titanic_tree准确率为：\n',score)
    #6.3 查看精确率、召回率、F1-score
    report = classification_report(y_test,y_predict,labels=[1,0],target_names=['获救','未获救'])
    print('精确率、召回率、F1-score报告：\n',report)
    #6.4roc_auc_score
    #6.4.1转化结果为0，1，用于二分类评估
    # y_true = np.where(y_test>2,1,0)
    ra_score = roc_auc_score(y_test,y_score=y_predict)
    print('roc_auc_score：',ra_score)

def titanic_forest():
    '''随机森林获取泰坦尼克获救人员数据'''
    df = pd.read_csv('data/titanic.csv')
    #2处理数据
    #2.1处理age缺失值
    df['age'].fillna(df['age'].mean(),inplace=True)
    x = df[['pclass','age','sex']]
    # x = df[['survived','age','pclass']]
    y = df['survived']
    # y = df['sex']
    #2.2转化为字典，由于age，pclass都是分类特征，最好转化成字典，然后用字典特征抽取
    x = x.to_dict(orient="records")
    #3数据集划分
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=6)
    #4字典特征抽取
    transtor = DictVectorizer()
    x_train = transtor.fit_transform(x_train)
    x_test = transtor.transform(x_test)
    #5随机森林预估器流程
    estimator= RandomForestClassifier()#n_estimators=200,max_depth=5)
    # 加入网格搜索与交叉验证
    # 参数准备
    # param_dict = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5,8,15,25,30]}
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)
    #6模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    # print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('titanic_forest准确率为：\n',score)
    #6.3 查看精确率、召回率、F1-score
    report = classification_report(y_test,y_predict,labels=[1,0],target_names=['获救','未获救'])
    print('精确率、召回率、F1-score报告：\n',report)
    #6.4roc_auc_score
    #6.4.1转化结果为0，1，用于二分类评估
    # y_true = np.where(y_test>2,1,0)
    ra_score = roc_auc_score(y_test,y_score=y_predict)
    print('roc_auc_score：',ra_score)
    #查看调优结果
    # 最佳参数
    # print('最佳参数：\n',estimator.best_params_)
    # #最佳结果
    # print('最佳结果:\n',estimator.best_score_)
    # #最佳估计器
    # print('最佳估计器：\n',estimator.best_estimator_)
    # #交叉验证结果
    # print('交叉验证结果:\n',estimator.cv_results_)

def titanic_logi():
    '''逻辑回归获取泰坦尼克获救人员数据'''
    df = pd.read_csv('data/titanic.csv')
    #2处理数据
    #2.1处理age缺失值
    df['age'].fillna(df['age'].mean(),inplace=True)
    x = df[['pclass','age','sex']]
    y = df['survived']
    #2.2转化为字典，由于age，pclass都是分类特征，最好转化成字典，然后用字典特征抽取
    x = x.to_dict(orient="records")
    #3数据集划分
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=6)
    #4字典特征抽取
    transtor = DictVectorizer()
    x_train = transtor.fit_transform(x_train)
    x_test = transtor.transform(x_test)
    #5逻辑回归预估器流程
    estimator= LogisticRegression()#n_estimators=200,max_depth=5)
    # estimator= LogisticRegressionCV()#n_estimators=200,max_depth=5)
    # 加入网格搜索与交叉验证
    # 参数准备
    # param_dict = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5,8,15,25,30]}
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)
    #6模型评估
    #方法1，直接比对
    y_predict = estimator.predict(x_test)
    # print('y_predict\n',y_predict)
    print('直接比对真实值和预测值\n',y_test == y_predict)
    #方法2，计算准确率
    score = estimator.score(x_test,y_test)
    print('titanic_forest准确率为：',score)
    #6.3 查看精确率、召回率、F1-score
    report = classification_report(y_test,y_predict,labels=[1,0],target_names=['获救','未获救'])
    print('精确率、召回率、F1-score报告：\n',report)
    #6.4roc_auc_score
    #6.4.1转化结果为0，1，用于二分类评估
    # y_true = np.where(y_test>2,1,0)
    ra_score = roc_auc_score(y_test,y_score=y_predict)
    print('roc_auc_score：',ra_score)
    #查看调优结果
    # #最佳参数
    # print('最佳参数：\n',estimator.best_params_)
    # #最佳结果
    # print('最佳结果:\n',estimator.best_score_)
    # #最佳估计器
    # print('最佳估计器：\n',estimator.best_estimator_)
    # #交叉验证结果
    # print('交叉验证结果:\n',estimator.cv_results_)

if __name__ == "__main__":
    # knn_iris()
    # knn_iris_gscv()
    nb_demo()
    # tree_iris()
    # titanic_tree()
    # titanic_forest()
    # titanic_logi()
    