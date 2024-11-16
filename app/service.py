"""This module contains all API endpoints."""

import os

from flask import jsonify, request
from flask import Blueprint, Response
from werkzeug.utils import secure_filename

from utils.constants import STORAGE_PATH
from utils.processing import prepare_data, decode_batch_predictions

from app.load_models import load_lstm_1, load_lstm_2, load_bilstm_1, load_bilstm_2

ai_model = Blueprint("ai_service", __name__)

LSTM_1, LSTM_2 = load_lstm_1(), load_lstm_2()
BILSTM_1, BILSTM_2 = load_bilstm_1(), load_bilstm_2()


def clear_storage():
    """
    Clear the storage directory.
    """
    directory = STORAGE_PATH
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))


@ai_model.route("/lstm1", methods=["POST"])
def predict_captcha_lstm_1():
    """
    Predict the CAPTCHA using model with 1 layer LSTM.

    Returns:
        A JSON response containing the predicted CAPTCHA text.
    """
    return predict_captcha("1 layer LSTM", "LSTM_1")


@ai_model.route("/lstm2", methods=["POST"])
def predict_captcha_lstm_2():
    """
    Predict the CAPTCHA using model with 2 layers LSTM.

    Returns:
        A JSON response containing the predicted CAPTCHA text.
    """
    return predict_captcha("2 layers LSTM", "LSTM_2")


@ai_model.route("/bilstm1", methods=["POST"])
def predict_captcha_bilstm_1():
    """
    Predict the CAPTCHA using model with 1 layer BiLSTM.

    Returns:
        A JSON response containing the predicted CAPTCHA text.
    """
    return predict_captcha("1 layer BiLSTM", "BILSTM_1")


@ai_model.route("/bilstm2", methods=["POST"])
def predict_captcha_bilstm_2():
    """
    Predict the CAPTCHA using model with 2 layers BiLSTM.

    Returns:
        A JSON response containing the predicted CAPTCHA text.
    """
    return predict_captcha("2 layers BiLSTM", "BILSTM_2")


def predict_captcha(model_name, model):
    """
    Predict the CAPTCHA using the specified model.

    Args:
        model_name (str): A string representing the model name.
        model (keras.models.Model): The model object to use for prediction.

    Returns:
        A JSON response containing the predicted CAPTCHA text.
    """
    file = request.files.get("file")
    uid = request.form.get("uid")

    if not file:
        return Response(response="[No CAPTCHA uploaded]", status=400)

    filename = secure_filename(file.filename)
    file_path = os.path.join(STORAGE_PATH, filename)
    file.save(file_path)

    prediction = globals()[model].predict(prepare_data(file_path))
    predicted_text = decode_batch_predictions(prediction)

    clear_storage()

    response_data = {"model": model_name, "prediction": str(*predicted_text)}

    if uid:
        response_data["uid"] = uid

    return jsonify(response_data)
