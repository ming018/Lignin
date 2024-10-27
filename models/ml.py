import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def linear_regression(X, y, X_pred):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X_pred)

def polynomial_regression(X, y, X_pred, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_pred_poly = poly.transform(X_pred)
    return model.predict(X_pred_poly)

def random_forest(X, y, X_pred):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model.predict(X_pred)

def gaussian_process(X, y, X_pred):
    kernel = RBF(length_scale=[10, 1], length_scale_bounds=(1e-2, 1e3))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
    model.fit(X, y)
    y_pred, y_std = model.predict(X_pred, return_std=True)
    return y_pred, y_std

def support_vector_regression(X, y, X_pred):
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X, y)
    return model.predict(X_pred)

def k_nearest_neighbors(X, y, X_pred, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model.predict(X_pred)


def plot_results(time, y_pred, y_std=None, model_name='Model'):
    plt.plot(time, y_pred, label=f'Predicted by {model_name}')
    if y_std is not None:
        plt.fill_between(time, y_pred - y_std, y_pred + y_std, alpha=0.2, label=f'{model_name} Uncertainty')




def compare_models(data):
    plt.figure(figsize=(12, 8))

    time = np.arange(data.shape[1])
    desired_temp = 325
    X = []
    y = []
    temperatures = [250, 300, 350, 400]
    for idx, t in enumerate(time):
        for i,temp in enumerate(temperatures):
            X.append([temp, t])
            y.append(data[i][idx])

    X_pred = np.array([[desired_temp, t] for t in time])

    # 선형 회귀
    y_pred = linear_regression(X, y, X_pred)
    plot_results(time, y_pred, model_name='Linear Regression')

    # 다항 회귀
    y_pred = polynomial_regression(X, y, X_pred, degree=3)
    plot_results(time, y_pred, model_name='Polynomial Regression (degree=3)')

    # 랜덤 포레스트
    y_pred = random_forest(X, y, X_pred)
    plot_results(time, y_pred, model_name='Random Forest')

    # 가우시안 프로세스
    # y_pred, y_std = gaussian_process(X, y, X_pred)
    # plot_results(time, y_pred, y_std, model_name='Gaussian Process')

    # 서포트 벡터 회귀
    y_pred = support_vector_regression(X, y, X_pred)
    plot_results(time, y_pred, model_name='SVR')

    # K-최근접 이웃
    y_pred = k_nearest_neighbors(X, y, X_pred, n_neighbors=5)
    plot_results(time, y_pred, model_name='K-Nearest Neighbors')

    #plot_grouped_bar_chart()

    # 실제 온도 데이터 시각화
    for i,temp in enumerate(temperatures):
        plt.plot(time, data[i], linestyle='--', label=f'Observed at {temp}K')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison of Machine Learning Models')
    plt.legend()
    plt.show()
