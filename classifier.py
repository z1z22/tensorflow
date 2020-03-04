from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def wine_nb():
    x,y =datasets.load_breast_cancer(return_X_y=True)

    x_train,x_test,y_train,y_test=train_test_split(x,y)

    tranfor = StandardScaler()
    x_train = tranfor.fit_transform(x_train)
    x_test = tranfor.transform(x_test)

    # estimator = KNeighborsClassifier(n_neighbors=10)
    # estimator = MultinomialNB()
    estimator = DecisionTreeClassifier()
    # estimator = RandomForestClassifier()

    estimator.fit(x_train,y_train)

    y_pred = estimator.predict(x_test)

    # print(y_test,'\n',y_pred)

    print('score:',estimator.score(x_test,y_test))

    print(classification_report(y_test,y_pred))
    #roc_auc_score 主要用于二分类问题
    ra_score = roc_auc_score(y_test,y_score=y_pred)
    print('roc_auc_score：',ra_score)


if __name__ == "__main__":
    wine_nb()