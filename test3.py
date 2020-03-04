from sklearn.datasets import load_iris,fetch_20newsgroups,fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# import sklearn
# print(sklearn.__version__)
def iris_kn():
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=22,)
    tranfer  = StandardScaler()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)


    estimator = KNeighborsClassifier()

    param ={'n_nerghbors':[1,3,5,7,9]}
    estimator= GridSearchCV(estimator,param_grid=param,cv=10)
    estimator.fit(x_train,y_train)

    score = estimator.score(x_test,y_test)
    print('score',score)

    print(estimator.best_estimator_)
    print(estimator.best_index_)
    print(estimator.best_params_)
    print(estimator.best_score_)


def iris_tree():
    x,y = load_iris(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 22)
    tranfer = MinMaxScaler()
    x_train=tranfer.fit_transform(x_train)
    x_test =tranfer.transform(x_test)

    estimator = DecisionTreeClassifier(criterion='gini',max_depth=6,)
    estimator.fit(x_train,y_train)
    score= estimator.score(x_test,y_test)

    print('tree_score',score)

def iris_forest():

    x,y = load_iris(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 22)
    tranfer = StandardScaler()
    x_train=tranfer.fit_transform(x_train)
    x_test =tranfer.transform(x_test)

    estimator = RandomForestClassifier(criterion='gini',max_depth=8,)
    estimator.fit(x_train,y_train)
    score= estimator.score(x_test,y_test)

    print('forest_score',score)

def nb_news():
    news = fetch_20newsgroups()
    # print(news.data)
    # print(news.target)
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,random_state = 22)

    tranfer = TfidfVectorizer()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)
    print(x_train)

    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    score = estimator.score(x_test,y_test)
    print('score:',score)
def nb_news():
    news = fetch_20newsgroups()
    # print(news.data)
    # print(news.target)
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,random_state = 22)

    tranfer = TfidfVectorizer()
    x_train = tranfer.fit_transform(x_train)
    x_test = tranfer.transform(x_test)
    print(x_train)

    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    score = estimator.score(x_test,y_test)
    print('score:',score)
def nb_news2():
    news = fetch_20newsgroups_vectorized()
    # print(news.data)

    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,random_state = 22)


    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    score = estimator.score(x_test,y_test)
    print('score:',score)

 



if __name__ == "__main__":
    iris_kn()
    # iris_tree()
    # iris_forest()
    # nb_news()
    # nb_news2()

    



