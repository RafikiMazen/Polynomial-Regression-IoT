from pandas import read_csv
from setuptools._vendor.six import print_
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import os
import operator




print("hello?")
os.chdir("../clustered_sensors/")
cluster = "0"
main_sensor = "0"
df = read_csv("./" + cluster + "/" + main_sensor +".csv")
df_b = read_csv("../Simulation/simulation_data/" + cluster +"/" +main_sensor + ".csv")
df_b['timestamp'] = pd.to_datetime(df_b['timestamp'])
df_b['timestamp']=(pd.to_numeric(df_b.timestamp))/10**11
cols = ['temp_max','temp_min','temp_avg','light_max','light_min','light_avg','humidity_min','humidity_max','humidity_avg']
data_cols = ['timestamp','light_avg','humidity_avg']
target_col=['temp_avg']
data_cols.append(target_col[0])
X = pd.DataFrame({},columns=data_cols)
# X[target_col[0]] = df_b[target_col[0]]
Y = pd.DataFrame({}, columns=target_col)
# X = pd.concat([X[data_cols], df[data_cols]], axis=0)
# Y = pd.concat([Y[target_col], df[target_col]], axis=0)
# # X.index = df['timestamp']
# # Y.index = df[target_col]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
def create_polynomial_regression_model(degree):
    "Creates a polynomial regression model for the given degree"

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))
    r2_train = r2_score(Y_train, y_train_predicted)

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    r2_test = r2_score(Y_test, y_test_predict)
    plt.scatter(Y_test, y_test_predict)
    # plt.show()


    # for c in X_test.columns:
    #     plt.scatter(X_test[c], Y_test, s=10)
    #     # sort the values of x before line plot
    #     sort_axis = operator.itemgetter(0)
    #     sorted_zip = sorted(zip(X_test, y_test_predict), key=sort_axis)
    #     xplot, yplot = zip(*sorted_zip)
    #     # plt.plot(xplot, yplot, color='m')
    #     plt.show()

    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")

    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
# degree = int(input())
# create_polynomial_regression_model(degree)
distances_df = read_csv("../Simulation/distances/" + cluster + "/" + main_sensor + ".csv")
distances_list = list(distances_df["sensor"].values)
for sensor in distances_list:
    df = read_csv("./" + cluster + "/" + str(int(sensor)) + ".csv")
    df_b = read_csv("../Simulation/simulation_data/" + cluster + "/" + str(int(sensor)) + ".csv")
    df_b['timestamp'] = pd.to_datetime(df_b['timestamp'])
    # df['timestamp'] = pd.to_numeric(df.timestamp)
    df_b['timestamp'] = (pd.to_numeric(df_b.timestamp)) / 10 ** 11
    target_column_in_X = X[target_col[0]]
    X = pd.concat([X[data_cols], df_b[data_cols]], axis=0)
    # X[target_col[0]] = pd.concat([target_column_in_X, df_b[target_col[0]]], axis=0).values
    Y = pd.concat([Y[target_col], df[target_col]], axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    degree = 1
    while(degree!=-1):
        create_polynomial_regression_model(degree)
        degree = int(input())