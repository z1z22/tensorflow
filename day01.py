from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer#字典特征抽取
from sklearn.feature_extraction.text import CountVectorizer#文本特征抽取方法1.对文本数据进行特征值化
from sklearn.feature_extraction.text import TfidfVectorizer#文本特征抽取方法2.自动找出关键词，并进行分类tf-idf
from sklearn.preprocessing import MinMaxScaler#做数据归一化处理
from sklearn.preprocessing import StandardScaler#做数据标准化处理
from sklearn.feature_selection import VarianceThreshold#过滤方差大于某个数的特征
from sklearn.decomposition import PCA#PCA降维
import jieba#中文分词处理
from scipy.stats import pearsonr#皮尔森相关系数
import numpy as np
import pandas as pd


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
    print('x_train',x_train.shape)
    print('x_test',x_test.shape)
    print('y_test',y_test.shape)
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
    print(data_new.toarray())
    print('特征名字:\n',tranfer.get_feature_names())

def cut_word(text):
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
    '''用tf-dif的方法进行文本特征抽取，tf(词频)*dif(词的权重对数)较之前的方法更科学更常用'''
    data='''当日新增治愈出院病例2742例，解除医学观察的密切接触者7650人，重症病例减少304例。
        截至3月2日24时，据31个省（自治区、直辖市）和新疆生产建设兵团报告，现有确诊病例30004例（其中重症病例6806例），累计治愈出院病例47204例，累计死亡病例2943例，累计报告确诊病例80151例，现有疑似病例587例。累计追踪到密切接触者664899人，尚在医学观察的密切接触者40651人。
        湖北新增确诊病例114例（武汉111例），新增治愈出院病例2410例（武汉1846例），新增死亡病例31例（武汉24例），现有确诊病例28216例（武汉24144例），其中重症病例6593例（武汉6020例）。累计治愈出院病例36167例（武汉23031例），累计死亡病例2834例（武汉2251例），累计确诊病例67217例（武汉49426例）。新增疑似病例64例（武汉62例），现有疑似病例434例（武汉316例）。
        累计收到港澳台地区通报确诊病例151例：香港特别行政区100例（出院36例，死亡2例），澳门特别行政区10例（出院8例），台湾地区41例(出院12例，死亡1例)。'''
    datalist = data.split('。')
    print(datalist)
    data_cut = []
    for sent in datalist:
        data_cut.append(cut_word(sent))
    # print(data_cut)
    #1.实例化转换器类
    tranfer=TfidfVectorizer()#参数stop_word=[]可以制定不进行分类抽取的词
    #2、调用fit_transform
    data_new = tranfer.fit_transform(data_cut)
    # print(data_new.toarray())
    # print('特征名字:\n',tranfer.get_feature_names())

    pca_tranfer = PCA(n_components=0.95)#小数代表百分比，整数代表维数
    data_pca = pca_tranfer.fit_transform(data_new.toarray())
    print(data_pca)


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

    1、实例化一个转换器类
    transfer = PCA(n_components=0.95)#小数代表百分比，整数代表维数

    2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    
    # datai = np.matrix(data).I #矩阵求逆
    # print(data.dot(datai))

if __name__ == '__main__':
    # dataset_demo()
    # dict_demo()
    # count_demo()
    # count_chinese_demo()
    # count_chinese_demo2()
    # tfidf_demo()
    # tfidf_demo2()
    # minmax_demo()
    # stand_demo()
    # variance_demo()
    pca_demo()