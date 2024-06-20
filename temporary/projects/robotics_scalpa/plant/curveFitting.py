from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

poly = Pipeline([
        ("poly", PolynomialFeatures(degree=3)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])

poly_ridge = Pipeline([
        ("poly", PolynomialFeatures(degree=20)),
        ("std_scaler", StandardScaler()),
        ("Ridge", Ridge(alpha=0.01))
    ])

poly_lasso = Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("std_scaler", StandardScaler()),
        ("Lasso", Lasso(alpha=0.1))
    ])

poly_elasticnet = Pipeline([
        ("poly", PolynomialFeatures(degree=20)),
        ("std_scaler", StandardScaler()),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5))
    ])

def  polynomial_fit(X: np.ndarray, Y: np.ndarray):

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    mode = 0
    flag_show = True

    if mode == 0:
        poly.fit(X, Y)
        Y_predict = poly.predict(X)
        X_row = X.reshape(X.shape[0])

        if flag_show:
            plt.scatter(X_row, Y)
            plt.plot(np.sort(X_row), Y_predict[np.argsort(X_row)], color='r')
            plt.show()

        return poly

    elif mode == 1:
        poly_ridge.fit(X, Y)
        Y_predict = poly_ridge.predict(X)
        X_row = X.reshape(X.shape[0])
        return poly_ridge

    elif mode == 2:
        poly_lasso.fit(X, Y)
        Y_predict = poly_lasso.predict(X)
        X_row = X.reshape(X.shape[0])
        return poly_lasso

    elif mode == 3:
        poly_elasticnet.fit(X, Y)
        Y_predict = poly_elasticnet.predict(X)
        X_row = X.reshape(X.shape[0])

        return poly_elasticnet
    pass


if __name__ == "__main__":
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)
    Y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    poly_reg = polynomial_fit(X, Y)
    Y_predict = poly_reg.predict(X)

    pass