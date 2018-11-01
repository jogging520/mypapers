from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import pandas as pd
from math import sqrt
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# usage :The Pyhton File is ARIMA prediction for 1 step



# Define Date function to tranfer Date to format"month,day,year"
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y')


def ARIMA_Predict(train, test, timestep=1):
    # 1. Divide X into trainning and testing data set
    history = [x for x in train]

    # 5. set prediction results
    predictions = list()
    print(history)

    # 6 . fit model for steps
    loop = len(test) / timestep
    for t in range(int(loop)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=timestep)
        yhat = output[0]
        # print('output:',output[0])
        for yhat_item in yhat:
            predictions.append(yhat_item)
        obs = test[t:t + timestep]
        for obs_item in obs:
            history.append(obs_item)
    return predictions


# main function
if __name__ == "__main__":
    # 1. Read data
    #series = read_csv('sp500_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    dataset = read_csv('sp500.csv', header=0, index_col=['trade-date'])
    # 2. Get Input as X
    X = dataset.values

    # 3. Divide X into trainning and testing data set
    size = int(len(X) * 0.7)
    train, test = X[0:size], X[size:]

    # 4. set time_step
    timestep = 1

    predictions = ARIMA_Predict(train, test, timestep)

    # 7 save result
    savetest = pd.DataFrame(test)
    savetest.to_csv('test.csv')
    save = pd.DataFrame(predictions)
    save.to_csv('Arima_step' + str(timestep) + '.csv')
    # 8 compute mse rmse mae
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # calulate MAE

    mae = mean_absolute_error(test, predictions)
    print('Test MAE: %.3f' % mae)
    # 9. plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
