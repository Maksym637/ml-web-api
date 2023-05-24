"""
This module contains functions for uploading different pre-trained models.
"""

from models.lstm_model import initialize_lstm_1, initialize_lstm_2
from models.bilstm_model import initialize_bilstm_1, initialize_bilstm_2

from utils.constants import (
    WEIGHTS_PATH_LSTM_1, WEIGHTS_PATH_LSTM_2,
    WEIGHTS_PATH_BILSTM_1, WEIGHTS_PATH_BILSTM_2,
)
from utils.processing import define_model

def load_lstm_1():
    """
    Load the pre-trained model with 1 layer LSTM.

    Returns:
        A defined model with loaded weights.
    """
    model = initialize_lstm_1()
    model.load_weights(WEIGHTS_PATH_LSTM_1)
    return define_model(model)

def load_lstm_2():
    """
    Load the pre-trained model with 2 layers LSTM.

    Returns:
        A defined model with loaded weights.
    """
    model = initialize_lstm_2()
    model.load_weights(WEIGHTS_PATH_LSTM_2)
    return define_model(model)

def load_bilstm_1():
    """
    Load the pre-trained model with 1 layer BiLSTM.

    Returns:
        A defined model with loaded weights.
    """
    model = initialize_bilstm_1()
    model.load_weights(WEIGHTS_PATH_BILSTM_1)
    return define_model(model)

def load_bilstm_2():
    """
    Load the pre-trained model with 2 layers BiLSTM.

    Returns:
        A defined model with loaded weights.
    """
    model = initialize_bilstm_2()
    model.load_weights(WEIGHTS_PATH_BILSTM_2)
    return define_model(model)
