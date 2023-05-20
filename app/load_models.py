from models.lstm_model import initialize_LSTM_1, initialize_LSTM_2
from models.bilstm_model import initialize_BiLSTM_1, initialize_BiLSTM_2

from utils.constants import (
    WEIGHTS_PATH_LSTM_1, WEIGHTS_PATH_LSTM_2,
    WEIGHTS_PATH_BiLSTM_1, WEIGHTS_PATH_BiLSTM_2,
)
from utils.processing import define_model

lstm_1, lstm_2 = initialize_LSTM_1(), initialize_LSTM_2()
bilstm_1, bilstm_2 = initialize_BiLSTM_1(), initialize_BiLSTM_2()

# lstm_1.load_weights(WEIGHTS_PATH_LSTM_1)
# defined_lstm_1 = define_model(lstm_1)

# lstm_2.load_weights(WEIGHTS_PATH_LSTM_2)
# defined_lstm_2 = define_model(lstm_2)

# bilstm_1.load_weights(WEIGHTS_PATH_BiLSTM_1)
# defined_bilstm_1 = define_model(bilstm_1)

bilstm_2.load_weights(WEIGHTS_PATH_BiLSTM_2)
defined_bilstm_2 = define_model(bilstm_2)
