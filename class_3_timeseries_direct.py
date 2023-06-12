import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def create_direct_data(data, feature, window_size, target_size):
    i = 1
    while i < window_size:
        data["{}_{}".format(feature, i)] = data[feature].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data[feature].shift(-window_size-i*4)
        i += 1
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
target_size = 3
data = create_direct_data(data, "co2", windows_size, target_size)
target = ["target_{}".format(i) for i in range(target_size)]
# # data = data.drop("time", axis=1)
x = data.drop(["time"] + target, axis=1)
y = data[target]
# split data
train_size = 0.8
num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]
#
#
regs = [LinearRegression() for _ in range(target_size)]
# # reg = RandomForestRegressor()
for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])
r2 = []
mae = []
mse = []
fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples*train_size)], data["co2_4"][:int(num_samples*train_size)], label="train")
ax.plot(data["time"][int(num_samples*train_size):], data["co2_4"][int(num_samples*train_size):], label="test")
for i, reg in enumerate(regs):
    y_predict = reg.predict(x_test)
    ax.plot(data["time"][int(num_samples * train_size):], y_predict, label="prediction_{}".format(i))
    r2.append(r2_score(y_test["target_{}".format(i)], y_predict))
    mae.append(mean_absolute_error(y_test["target_{}".format(i)], y_predict))
    mse.append(mean_squared_error(y_test["target_{}".format(i)], y_predict))

print("R2 score {}".format(r2))
print("Mean absolute error {}".format(mae))
print("Mean squared error {}".format(mse))

ax.set_xlabel("Year")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()
