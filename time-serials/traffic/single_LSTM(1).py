import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics

def loaddata(path, lags):
    df = pd.read_csv(path, index_col=None, header=None)
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df.values.astype('float32')
    data = scaler.fit_transform(values)
    dataset = list()
    for i in range(lags, len(data)):
        dataset.append(data[i - lags: i + 1])

    dataset = np.array(dataset)
    print('dataset',dataset.shape)
    t = int(len(dataset)*0.8)
    train = dataset[:t, :]
    test = dataset[t:, :]
    np.random.shuffle(train)
    print('train', train.shape)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    print('X_train',X_train.shape)
    print('y_train',y_train.shape)
    return X_train, y_train, X_test, y_test, scaler

def lstm(X_train, y_train, X_test, y_test, scaler):
    # build model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True, activation='sigmoid'))
    model.add(LSTM(10, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # compile and fit model
    adam = Adam(lr=0.003)
    model.compile(loss='mse', optimizer=adam, metrics=['mape'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=False)
    # plot loss
    # plt.figure()
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    # prediction

    print('X_test',X_test.shape)
    pre = model.predict(X_test)
    print('pre',pre.shape)
    pre = scaler.inverse_transform(pre)
    true = scaler.inverse_transform(y_test)
    plt.figure()
    plt.plot(pre[:], 'r--', label='Predicted Values')
    plt.plot(true[:], 'b', label='Observed Values')
    plt.legend()
    plt.show()

    return pre, true

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % np.sqrt(mse))
    print('r2:%f' % r2)

if __name__ == '__main__':
    path = 'HI7052a.txt'
    lags = 7
    feature = 1
    X_train, y_train, X_test, y_test, scaler = loaddata(path, lags)
    X_train = X_train.reshape(X_train.shape[0], lags, feature)
    X_test = X_test.reshape(X_test.shape[0], lags, feature)
    pre, true = lstm(X_train, y_train, X_test, y_test, scaler)
    eva_regress(true, pre)
    print(pre.shape)
    print(true.shape)