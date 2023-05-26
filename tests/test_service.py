"""This module contains tests for the API service."""

import pytest
from run import app

client = app.test_client()

TEST_DATA_POSITIVE = [
    ("captcha-1.png", "00fggp"),
    ("captcha-2.png", "00jphx"),
    ("captcha-3.png", "01uk0d"),
    ("captcha-4.png", "058zms"),
    ("captcha-5.png", "004mqe")
]

TEST_DATA_NEGATIVE = [
    ("/lstm1", "[No CAPTCHA uploaded]"),
    ("/lstm2", "[No CAPTCHA uploaded]"),
    ("/bilstm1", "[No CAPTCHA uploaded]"),
    ("/bilstm2", "[No CAPTCHA uploaded]")
]

@pytest.mark.parametrize("image,expected_result", TEST_DATA_POSITIVE)
def test_predict_lstm_1_positive(image, expected_result):
    """
    Test positive prediction for a model with 1 layer LSTM.

    Args:
        image (str): Path to the captcha image file.
        expected_result (str): Expected prediction result.
    """
    endpoint = "/lstm1"
    captcha = f"./data/{image}"
    data = {"file": (open(captcha, "rb"), image)}

    response = client.post(
        endpoint,
        data=data,
        buffered=True,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] == expected_result

@pytest.mark.parametrize("image,expected_result", TEST_DATA_POSITIVE)
def test_predict_lstm_2_positive(image, expected_result):
    """
    Test positive prediction for a model with 2 layers LSTM.

    Args:
        image (str): Path to the captcha image file.
        expected_result (str): Expected prediction result.
    """
    endpoint = "/lstm2"
    captcha = f"./data/{image}"
    data = {"file": (open(captcha, "rb"), image)}

    response = client.post(
        endpoint,
        data=data,
        buffered=True,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] == expected_result

@pytest.mark.parametrize("image,expected_result", TEST_DATA_POSITIVE)
def test_predict_bilstm_1_positive(image, expected_result):
    """
    Test positive prediction for a model with 1 layer BiLSTM.

    Args:
        image (str): Path to the captcha image file.
        expected_result (str): Expected prediction result.
    """
    endpoint = "/bilstm1"
    captcha = f"./data/{image}"
    data = {"file": (open(captcha, "rb"), image)}

    response = client.post(
        endpoint,
        data=data,
        buffered=True,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] == expected_result

@pytest.mark.parametrize("image,expected_result", TEST_DATA_POSITIVE)
def test_predict_bilstm_2_positive(image, expected_result):
    """
    Test positive prediction for a model with 2 layers BiLSTM.

    Args:
        image (str): Path to the captcha image file.
        expected_result (str): Expected prediction result.
    """
    endpoint = "/bilstm2"
    captcha = f"./data/{image}"
    data = {"file": (open(captcha, "rb"), image)}

    response = client.post(
        endpoint,
        data=data,
        buffered=True,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.get_json()["prediction"] == expected_result

@pytest.mark.parametrize("endpoint,expected_result", TEST_DATA_NEGATIVE)
def test_predict_model_negative(endpoint, expected_result):
    """
    Verifies the negative scenario. A case when image is not uploaded.

    Args:
        endpoint (str): Endpoint of a specific model.
        expected_result (str): The expected error message.
    """
    response = client.post(endpoint)

    assert response.status_code == 400
    assert response.text == expected_result
