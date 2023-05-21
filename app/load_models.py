from models.lstm_model import initialize_LSTM_1, initialize_LSTM_2
from models.bilstm_model import initialize_BiLSTM_1, initialize_BiLSTM_2

from utils.constants import (
    WEIGHTS_PATH_LSTM_1, WEIGHTS_PATH_LSTM_2,
    WEIGHTS_PATH_BiLSTM_1, WEIGHTS_PATH_BiLSTM_2,
)
from utils.processing import define_model

def load_lstm_1():
    model = initialize_LSTM_1()
    model.load_weights(WEIGHTS_PATH_LSTM_1)
    return define_model(model)

def load_lstm_2():
    model = initialize_LSTM_2()
    model.load_weights(WEIGHTS_PATH_LSTM_2)
    return define_model(model)

def load_bilstm_1():
    model = initialize_BiLSTM_1()
    model.load_weights(WEIGHTS_PATH_BiLSTM_1)
    return define_model(model)

def load_bilstm_2():
    model = initialize_BiLSTM_2()
    model.load_weights(WEIGHTS_PATH_BiLSTM_2)
    return define_model(model)
