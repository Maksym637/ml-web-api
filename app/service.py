import os

from flask import jsonify, request
from flask import Blueprint, Response
from werkzeug.utils import secure_filename

from utils.constants import STORAGE_PATH
from utils.processing import prepare_data, decode_batch_predictions

from app.load_models import (
    load_lstm_1, load_lstm_2,
    load_bilstm_1, load_bilstm_2
)

ai_model = Blueprint("ai_service", __name__)

LSTM_1, LSTM_2 = load_lstm_1(), load_lstm_2() 
BiLSTM_1, BiLSTM_2 = load_bilstm_1(), load_bilstm_2()

def clear_storage():
    directory = STORAGE_PATH
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

@ai_model.route("/lstm1", methods=["POST"])
def predict_captcha_lstm_1():
    file = request.files["image"]

    if not file:
        return Response(response="[No CAPTCHA uploaded]", status=400)

    filename = secure_filename(file.filename)
    file_path = os.path.join(STORAGE_PATH, filename)
    file.save(file_path)

    prediction = LSTM_1.predict(prepare_data(file_path))
    predicted_text = decode_batch_predictions(prediction)

    clear_storage()

    return jsonify({"prediction": predicted_text})

@ai_model.route("/lstm2", methods=["POST"])
def predict_captcha_lstm_2():
    file = request.files["image"]

    if not file:
        return Response(response="[No CAPTCHA uploaded]", status=400)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(STORAGE_PATH, filename)
    file.save(file_path)

    prediction = LSTM_2.predict(prepare_data(file_path))
    predicted_text = decode_batch_predictions(prediction)

    clear_storage()

    return jsonify({"prediction": predicted_text})

@ai_model.route("/bilstm1", methods=["POST"])
def predict_captcha_bilstm_1():
    file = request.files["image"]

    if not file:
        return Response(response="[No CAPTCHA uploaded]", status=400)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(STORAGE_PATH, filename)
    file.save(file_path)

    prediction = BiLSTM_1.predict(prepare_data(file_path))
    predicted_text = decode_batch_predictions(prediction)

    clear_storage()

    return jsonify({"prediction": predicted_text})

@ai_model.route("/bilstm2", methods=["POST"])
def predict_captcha_bilstm_2():
    file = request.files["image"]

    if not file:
        return Response(response="[No CAPTCHA uploaded]", status=400)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(STORAGE_PATH, filename)
    file.save(file_path)

    prediction = BiLSTM_2.predict(prepare_data(file_path))
    predicted_text = decode_batch_predictions(prediction)

    clear_storage()

    return jsonify({"prediction": predicted_text})
