import pandas as pd
import pandas_datareader as web
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from flask import Flask
from flask_cors import CORS

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM


def stock_model(stock_name, train_start_date, train_end_date):
    train_data = web.DataReader(stock_name, "yahoo", train_start_date, train_end_date)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(train_data["Close"].values.reshape(-1, 1))

    prediction_days = 60

    # Prepare training data
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=100, batch_size=64)

    # Save model
    path = "saved_model/LSTM_model"
    model.save(path)
    return path


def stock_test(model_path, stock_name, test_start_date, test_end_date, prediction_days):
    # Prepare test data
    test_data = web.DataReader(stock_name, "yahoo", test_start_date, test_end_date)
    test_actual_price = test_data["Close"].values
    train_data = web.DataReader(stock_name, "yahoo", dt.datetime(2017, 1, 1), dt.datetime(2022, 1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(train_data["Close"].values.reshape(-1, 1))

    dataset = pd.concat((train_data["Close"], test_data["Close"]), axis=0)
    test_input = dataset[len(dataset) - len(test_data) - prediction_days:].values
    test_input = test_input.reshape(-1, 1)
    test_input = scaler.transform(test_input)

    # Run model with test data
    x_test = []
    for x in range(prediction_days, len(test_input)):
        x_test.append(test_input[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # use built model from path param
    model = load_model(model_path)
    predict_price = model.predict(x_test)
    predict_price = scaler.inverse_transform(predict_price)

    # Check error with test data
    rmse = np.sqrt(np.mean(predict_price - test_actual_price) ** 2)
    print("RMSE is " + str(rmse))  # RMSE is 0.5634308591413726

    plt.plot(predict_price)
    plt.plot(test_actual_price)
    plt.show()


def stock_predict(model_path, stock_name, duration, prediction_days):
    dataset = web.DataReader(stock_name, "yahoo", dt.datetime(2022, 1, 1), dt.datetime.now())
    # Get past 30 days real data
    predict_actual_dataset = dataset[-duration:]
    predict_actual_price = predict_actual_dataset["Close"].values

    # Get past 60 days feature data
    predict_input = dataset[-(prediction_days+duration):-duration]
    predict_input = predict_input["Close"].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(dataset["Close"].values.reshape(-1, 1))
    predict_input = predict_input.reshape(-1, 1)
    predict_input = scaler.transform(predict_input)

    # Predict with sliding window
    model = load_model(model_path)
    tmp_input = predict_input.flatten()
    tmp_input = tmp_input.tolist()
    predict_result = []
    for i in range(0, duration):
        if (len(tmp_input) <= prediction_days):
            predict_input = predict_input.reshape((1, prediction_days, 1))
            predict_output = model.predict(predict_input)
            tmp_input.extend(predict_output[0].tolist())
            predict_result.extend(predict_output.tolist())

        else:
            predict_input = np.array(tmp_input[1:])
            predict_input = predict_input.reshape(1, -1)  # TODO:
            predict_input = predict_input.reshape((1, prediction_days, 1))
            predict_output = model.predict(predict_input)
            tmp_input.extend(predict_output[0].tolist())
            tmp_input = tmp_input[1:]
            predict_result.extend(predict_output.tolist())

    predict_price = scaler.inverse_transform(predict_result)

    predict_actual_result = predict_actual_dataset.filter(["Close"])
    predict_actual_result["Prediction"] = predict_price

    return predict_actual_result


# flask api
app = Flask(__name__)
CORS(app)

@app.route("/")
def stock_prediction_api():
    MODEL_PATH = "saved_model/LSTM_model"
    predict_actual_result = stock_predict(MODEL_PATH, "AAPL", 30, 60)
    print(predict_actual_result)
    return predict_actual_result.to_json(orient="index", date_format="iso")

if __name__ == '__main__':
    # Build and test model
    #model_path = stock_model("AAPL", dt.datetime(2012, 1, 1), dt.datetime(2022, 1, 1))
    #stock_test(model_path, "AAPL", dt.datetime(2022, 1, 1), dt.datetime.now(), 60)
    #stock_predict(model_path, "AAPL", 30, 60)

    # Test
    #stock_test("saved_model/LSTM_model", "GOOGL", dt.datetime(2022, 1, 1), dt.datetime.now(), 60)
    #stock_predict("saved_model/LSTM_model", "AAPL", 30, 60)

    app.run(host="0.0.0.0", port=80, debug = True)