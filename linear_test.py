from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor,LinearRegression,Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def linner_demo():
    x,y = make_regression(n_samples=100,n_features=1,n_targets=1,noise=20 )
    # print(dia)

    # print(x,y)
    # x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=25)

    # transfor = StandardScaler()
    # x_train = transfor.fit_transform(x_train)
    # x_test = transfor.transform(x_test)
    # estimator = SGDRegressor()
    # estimator = LinearRegression()
    estimator = Ridge()
    estimator.fit(x,y)
    #评估模型
    print(estimator.coef_)
    print(estimator.intercept_)
    y_predict = estimator.predict(x)
    print(y_predict,y)
    error = mean_squared_error(y,y_pred=y_predict)
    print('error',error)
    plt.scatter(x,y,)
    plt.scatter(x,y_predict)
    plt.show()





if __name__ == "__main__":
    linner_demo()
