from random import randint
from math import sqrt
from numpy import array
import numpy
import numpy as np
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as  pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import concatenate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# function:convert series to supervised learning（input,output）
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# function: plot function
def plot_results(predicted_data, true_data):
    fig = pyplot.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    pyplot.plot(predicted_data, label='Prediction')
    pyplot.legend()
    pyplot.show()

# function:  generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# function: Seq2Seq network returns train, inference_encoder and inference_decoder models
def build_model(train_x, train_y, n_input):
	# define parameters
	verbose, epochs, batch_size = 2, 3, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	# train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	print(train_x.shape)
	print(train_y.shape)
	print(n_outputs)
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

if __name__ == "__main__":
	# 1. load dataset
	dataset = read_csv('sp500.csv', header=0, index_col=0)
	values = dataset.values

	# 2. ensure all data is float
	values = values.astype('float32')

	# 3. normalize features(0-1)
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	# 4. set time_step
	time_step = 2

	# 5. frame as supervised learning(train_data -> predict_data )
	reframed = series_to_supervised(scaled, time_step, time_step)
	values = reframed.values

	# 6. set train_data and test_data set
	train_len = int(len(values) * 0.7)
	train = values[:train_len, :]
	test = values[train_len:, :]

	# 7. split into input and outputs(the length of input equals the length of output)
	train_X, train_Y = train[:, :-time_step], train[:, -time_step:]
	test_X, test_Y = test[:, :-time_step], test[:, -time_step:]
	# 8. reshape reshape input to be 3D[samples, time steps, features]
	trainX = numpy.reshape(train_X, (train_X.shape[0],time_step,1))
	trainY = numpy.reshape(train_Y, (train_X.shape[0],time_step,1))
	testX =  numpy.reshape(test_X, (test_X.shape[0],time_step,1))
	testY =  numpy.reshape(test_Y, (test_X.shape[0],time_step,1))
	print(trainX.shape, trainY.shape, testX.shape, testY.shape)

	# 9 configure problem
	n_steps_in = time_step
	n_steps_out = time_step
	# 10 fit model
	model = build_model(trainX, trainY, n_steps_in)
	#12   history is a list of data
	history = [x for x in trainX]
	#13  make prediction
	predictions = list()
	for i in range(testX.shape[0]):
		# predict the week
		yhat_sequence = forecast(model, history, n_steps_in)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(testX[i,::])
	# evaluate predictions days for each week
	predictions = array(predictions)
	predictions = predictions.reshape(predictions.shape[0],predictions.shape[1])


	#14 invert scaling for forecast
	inv_yhat = scaler.inverse_transform(predictions)
	inv_test = scaler.inverse_transform(test_Y)

   # 15 get prediction from yhat
	predict_result = list()
	test_result = list()
	t = 0
	while t < (test_X.shape[0]):
		for yhat_item in inv_yhat[t]:
			print(yhat_item)
			predict_result.append(yhat_item)
		for test_item in inv_test[t]:
			test_result.append(test_item)
		t = t + time_step
	save = pd.DataFrame(predictions)
	save.to_csv('LSTM_step' + str(time_step) + '.csv')
	print(len(predict_result))
	print(len(test_result))
	# 13 plot result
	plot_results(predict_result, test_result)
	pyplot.show()
	print('predictions', len(predict_result))
	print('test', len(test_result))

	# 16.0 calculate RMSE
	rmse = sqrt(mean_squared_error(predict_result, test_result))
	print('Test RMSE: %.3f' % rmse)
	# 16.1 calculate MSE
	error = mean_squared_error(predict_result, test_result)
	print('Test MSE: %.3f' % error)
	# 16.2  calulate MAE
	mae = mean_absolute_error(predict_result, test_result)
	print('Test MAE: %.3f' % mae)
   # pyplot.plot(inv_yhat)

