from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as  pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
##usage :The Pyhton File is BP NN net prediction for 1 step

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

# function:main function
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
    time_step =1

    # 5. frame as supervised learning(train_data -> predict_data )
    reframed = series_to_supervised(scaled, time_step, time_step)
    values = reframed.values

    # 6. set train_data and test_data set
    train_len = int(len(values)*0.7)
    train = values[:train_len, :]
    test = values[train_len:, :]

    # 7. split into input and outputs(the length of input equals the length of output)
    train_X, train_y = train[:, :-time_step], train[:, -time_step:]
    test_X, test_y = test[:, :-time_step], test[:, -time_step:]

    # 8. design network and  BP begins
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=time_step,activation='relu'))
    model.add(Dense(time_step))
    model.compile(loss='mae', optimizer='adam')
    # 9. fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=14, verbose=2, shuffle=False)

    #10. make a prediction
    yhat = model.predict(test_X)
    #print('yhat shape',yhat.shape)
    # 11. Transfer (0-1) to real scale
    inv_yhat = scaler.inverse_transform(yhat)
    #inv_yhat = inv_yhat[:, 0:time_step]
    inv_test = scaler.inverse_transform(test_y)

    print('inver_test', inv_test.shape)
    # 12 get prediction from yhat
    predictions = list()
    test_result = list()
    t = 0
    while t < (test_X.shape[0]):
        for yhat_item in inv_yhat[t]:
            predictions.append(yhat_item)
        for test_item in inv_test[t]:
            test_result.append(test_item)
        t = t + time_step
    save = pd.DataFrame(predictions)
    save.to_csv('BP_step'+str(time_step)+'.csv')
    print(len(predictions))
    print(len(test_result))
    # 13 plot result
    plot_results(predictions, test_result)
    pyplot.show()
    # print('predictions',len(predictions))
    # print('test', len(test_result))

    # 14.0 calculate RMSE
    rmse = sqrt(mean_squared_error(predictions, test_result))
    print('Test RMSE: %.3f' % rmse)

    error = mean_squared_error(predictions, test_result)
    print('Test MSE: %.3f' % error)
    #14.1  calculate RMSE
    rmse = sqrt(mean_squared_error(predictions, test_result))
    print('Test RMSE: %.3f' % rmse)
    #14.2  calulate MAE

    mae = mean_absolute_error(predictions, test_result)
    print('Test MAE: %.3f' % mae)
    #pyplot.plot(inv_yhat)

