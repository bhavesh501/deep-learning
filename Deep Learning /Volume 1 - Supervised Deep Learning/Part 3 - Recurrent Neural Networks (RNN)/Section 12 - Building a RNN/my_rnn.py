import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2]

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#getting the inputs and outputs for 1 timestep
x_train = training_set[0:1257]
y_train = training_set[1:1258]

"""#creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set[i-60 : i, 0 ])
    y_train.append(training_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)"""

#reshaping the input
x_train = np.reshape(x_train,(1257,1,1))

#Initialising the RNN 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()

#adding input layer and LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#adding the output layer
regressor.add(Dense(units = 1))

#compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

#fitting the RNN to training_set
regressor.fit(x_train, y_train, batch_size = 32, epochs = 200)

#making the predictions
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#visualising the results
plt.plot(real_stock_price, color = 'red', label = 'real_stock_price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predicted_stock_price')
plt.title('predicted stock price of google')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()