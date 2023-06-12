import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def create_recursive_data(data, feature, window_size):
    i = 1
    while i < window_size:
        data["{}_{}".format(feature, i)] = data[feature].shift(-i)
        i += 1
    data["target"] = data[feature].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("co2.csv")
# print(data.isna().sum())
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# plt.show()
windows_size = 5
data = create_recursive_data(data, "co2", 5)
target = "target"
# data = data.drop("time", axis=1)
x = data.drop([target, "time"], axis=1)
y = data[target]
# split data
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]


reg = LinearRegression()
# reg = RandomForestRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("R2 score: {}".format(r2_score(y_test, y_predict)))
print("Mean absolute error: {}".format(mean_absolute_error(y_test, y_predict)))
print("Mean squared error: {}".format(mean_squared_error(y_test, y_predict)))

fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples*train_size)], data["co2_4"][:int(num_samples*train_size)], label="train")
ax.plot(data["time"][int(num_samples*train_size):], data["co2_4"][int(num_samples*train_size):], label="test")
ax.plot(data["time"][int(num_samples*train_size):], y_predict, label="prediction")
ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()
