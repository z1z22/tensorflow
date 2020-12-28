from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer#字典特征抽取
from sklearn.feature_extraction.text import CountVectorizer#文本特征抽取方法1.对文本数据进行特征值化
from sklearn.feature_extraction.text import TfidfVectorizer#文本特征抽取方法2.自动找出关键词，并进行分类tf-idf
from sklearn.preprocessing import MinMaxScaler#做数据归一化处理
from sklearn.preprocessing import StandardScaler#做数据标准化处理
from sklearn.feature_selection import VarianceThreshold#过滤方差大于某个数的特征
from sklearn.decomposition import PCA#PCA降维
from sklearn.cluster import KMeans
import jieba#中文分词处理
from scipy.stats import pearsonr#皮尔森相关系数
import numpy as np
import pandas as pd
from collections import Counter

import re


def dataset_demo():
    '''调用数据集及数据集的划分'''
    iris =load_iris()
    print('鸢尾花数据集：\n',iris)
    print('查看属性：\n',iris.DESCR)
    print('查看特征名字：\n',iris.feature_names)
    print('查看目标值名字：\n',iris.target_names)
    print('查看数据：\n',iris.data)
    #数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2)
    print('x_train shape',x_train.shape)
    print('x_test shape',x_test.shape)
    print('y_test shape',y_test.shape)
    print('y_train',y_train)


def dict_demo():
    '''字典特征抽取DictVectorizer'''
    data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'武汉','temperature':30}]
    #1、实例化一个转化类
    tranfer=DictVectorizer(sparse=False)#sparse稀疏矩阵
    #2.调用fit_transform
    data_new= tranfer.fit_transform(data)
    print('data_new:\n',data_new)
    print('特征名字:\n',tranfer.get_feature_names())
    new_df= pd.DataFrame(data_new,columns=tranfer.get_feature_names())
    print(new_df)

def search_stopwords(string):
    stopwords = []
    with open('data/stopwords/stopword_ch_en.txt') as f:
        lines = f.readlines()
    for word in lines:
        if word.strip() in string:
            stopwords.append(word.strip())
    return stopwords    

def count_demo():
    '''文本特征抽取CountVectorizer,统计样本中特征词的个数'''
    data=['life is short, i like like python','life is too long,i dislike python']
    #1.实例化转换器类
    tranfer=CountVectorizer(stop_words=['is','too'])#参数stop_word=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data)
    print(data_new.toarray())
    print('特征名字:\n',tranfer.get_feature_names())

def count_chinese_demo():
    '''中文文本特征抽取CountVectorizer,注意要进行分词，统计样本中特征词的个数'''
    data=['武汉 必胜','武汉 肺炎 不是 事']
    #1.实例化转换器类
    tranfer=CountVectorizer()#参数stop_word=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data)
    print(data_new)
    print(data_new.toarray())#toarray将稀疏矩阵转化为普通矩阵存储
    print('特征名字:\n',tranfer.get_feature_names())

def cut_word(text):
    """使用jieba对文本进行分词操作"""
    return ' '.join(list(jieba.cut(text)))

def count_chinese_demo2():
    '''中文文本特征抽取CountVectorizer,用到自动分词jieba，统计样本中特征词的个数'''
    data=['泡影','两个自由的水泡,从梦海深处升起……','朦朦胧胧的银雾,在微风中散去','我象孩子一样,紧拉住渐渐模糊的你','徒劳的要把泡影,带回现实的陆地']
    data_cut = []
    for sent  in data:
        data_cut.append(cut_word(sent))
    print(data_cut)
    #1.实例化转换器类
    tranfer=CountVectorizer()#参数stop_word=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data_cut)
    print(data_new.toarray())
    print('特征名字:\n',tranfer.get_feature_names())

def txt_demo():
    '''将txt转化为向量'''
    stopwords = []
    # with open('data/stopwords/stopword_ch_en.txt') as f:
    #     lines = f.readlines()
    # for word in lines:
    #     stopwords.append(word.strip())

    with open('data/tiaopi.txt', 'r') as f:
        txt = f.readlines()
    txt_cut_list = []
    stopwords_list = []
    for line in txt:
        stopwords_list += search_stopwords(line)
        txt_cut_list.append(cut_word(line))
    stopwords = list(set(stopwords_list))
    print ('stopwords: ', stopwords)
    print('-------------------------------------')
        
    print('语句量:  ',len(txt_cut_list))

    # 参数stop_word=[]可以制定不进行分类抽取的词
    tranfer = TfidfVectorizer(stop_words=stopwords)
    #2、调用fit_transform
    data_new = tranfer.fit_transform(txt_cut_list)
    print('data_after_tfidf: ' ,data_new.toarray().shape)
    print('特征名字:  ', len(tranfer.get_feature_names()))
    
    pca_tranfer = PCA(n_components=0.9)  # 小数代表百分比，整数代表维数
    data_pca = pca_tranfer.fit_transform(data_new.toarray())
    # print(data_pca)
    print('data_pca: ', data_pca.shape)
    estimator = KMeans(n_clusters=10)
    estimator.fit(data_pca)

    y_predict = estimator.predict(data_pca)
    print(len(y_predict))
    print(y_predict)
    print(Counter(y_predict))



def tfidf_demo():
    '''用tf-dif的方法进行文本特征抽取，tf(词频)*dif(词的权重对数)较之前的方法更科学更常用'''
    data=['泡影','两个自由的水泡,从梦海深处升起……','朦朦胧胧的银雾,在微风中散去','我象孩子一样,紧拉住渐渐模糊的你','徒劳的要把泡影,带回现实的陆地']
    data_cut = []
    for sent  in data:
        data_cut.append(cut_word(sent))
    print(data_cut)
    #1.实例化转换器类
    tranfer=TfidfVectorizer()#参数stop_word=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data_cut)
    print(data_new.toarray())
    print('特征名字:\n',tranfer.get_feature_names())


def tfidf_demo2():
    '''
    用tf-dif的方法进行文本特征抽取，tf(词频)*dif(词的权重对数)较之前的方法更科学更常用
    '''
    data = '''近日，司法部、国家移民管理局在京联合召开《中华人民共和国外国人永久居留管理条例》（征求意见稿）征求意见座谈会。部分企事业单位人员、社区干部、居民和专家学者应邀参加会议，对《条例》（征求意见稿）进行深入讨论，发表意见建议。
    与会人员认为，通过赋予外国人永久居留资格吸引人才、专业人士和域外资本参与本国建设，促进经济社会发展，是许多国家在发展进程中的普遍做法。为了进一步规范和改进外国人永久居留审批管理工作，适应深化改革扩大开放、加快建设社会主义现代化国家的需要，促进中外交往，我国有必要在现行制度和实践的基础上制订外国人永久居留管理条例。
    与会人员指出，《条例》征求意见以来，社会高度关注，有的担心有关资格、条件设计是否合理，会不会出现大量境外人员挤占国内就业岗位和社会公共福利资源问题，也有的反映一些规定过于原则、不够细化，顾虑实施后出现管理漏洞、难以监督问题，与会人员建议结合我国国情以及国际有益做法，进一步评估论证，完善优化相关制度设计，使申请永久居留的资格、条件和程序更周延、更严密。
    司法部有关负责人表示，《条例》目前尚处于向社会征求意见阶段，我们将按照科学立法、民主立法、依法立法的要求，对公众所提的意见建议，认真深入予以研究。《条例》在充分吸纳公众意见、进一步修改完善之前不会仓促出台。
    国家移民管理局有关负责人表示，高度重视公众对完善境内外国人管理工作的关切，在依法保护广大中外出入境人员合法权益的同时，将进一步强化对非法入境、非法滞留人员检查、遣返力度，依法严肃查处相关违法犯罪活动，维护正常的出入境秩序。'''
    stopwords = search_stopwords(data)
    print('stopwords:\n',stopwords)

    datalist = data.split('。')
    # print(datalist)
    data_cut = []
    for sent in datalist:
        data_cut.append(cut_word(sent))
    # print(data_cut)

    #1.实例化转换器类
    tranfer = TfidfVectorizer(stop_words=stopwords)  # 参数stop_words=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data_cut)
    # print(data_new.toarray())
    print('特征名字:\n',tranfer.get_feature_names())

    pca_tranfer = PCA(n_components=0.95)#小数代表百分比，整数代表维数
    data_pca = pca_tranfer.fit_transform(data_new.toarray())
    print(data_pca)
    print(data_pca.shape)


def minmax_demo():
    '''归一化预处理，更多的用在小的整齐的数据中'''
    #1、获取数据
    data=pd.read_csv('data/china.csv')
    data = data.iloc[:,1:8].dropna()
    print(data)
    #2、实例化一个转化器类
    tranfer=MinMaxScaler()
    #MinMaxScaler(feature_range=(0,1), copy=True)
    #3、调用fit_transform
    data_new = tranfer.fit_transform(data)
    print(data_new)

def stand_demo():
    '''标准化预处理处理数据,较归一化更少受到空值等噪声干扰，较归一化常用'''
    #1、获取数据
    data=pd.read_csv('data/china.csv')
    data = data.iloc[:,1:8].dropna()
    print(data)
    #2、实例化一个转化器类
    tranfer=StandardScaler()
    #StandardScaler(copy=True, with_mean=True, with_std=True)
    #3、调用fit_transform
    data_new = tranfer.fit_transform(data)
    print(data_new)

def variance_demo():
    '''降维：过滤低方差数据'''
    #1、获取数据
    data = pd.read_csv('data/factor_returns.csv')
    data = data.iloc[:,1:-2]
    print(data)
    #2、实例化一个转换器类
    transfer = VarianceThreshold(threshold=5)
    #3、调用fit_transform
    new_data = transfer.fit_transform(data)
    print('new_data',new_data)
    print(new_data.shape)

    # 计算某两个变量之间的相关系数
    # 方法一:
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pe_ratio与pb_ratio相关系数：\n", r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("revenue与total_expense之间的相关性：\n", r2)
    #方法二:
    # 用pandas函数corr()，计算所有列的相关性
    print(data.corr())
    
def pca_demo():
    '''PCA降维，也叫主成分分析，降低数据的维度，并保持原有数据的特征'''
    
    data = np.random.random(49).reshape([7,7])
    print(data)

    # 1、实例化一个转换器类
    transfer = PCA(n_components=0.95)#小数代表百分比，整数代表维数

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    
    # datai = np.matrix(data).I #矩阵求逆
    # print(data.dot(datai))

if __name__ == '__main__':
    dataset_demo()
    # dict_demo()
    # count_demo()
    # count_chinese_demo()
    # count_chinese_demo2()
    # txt_demo()
    # tfidf_demo()
    # tfidf_demo2()
    # minmax_demo()
    # stand_demo()
    # variance_demo()
    # pca_demo()
