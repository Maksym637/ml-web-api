"""
Start the Flask application.

This script starts the Flask application with the following configurations:
    - Debug mode is enabled for easier development and troubleshooting.
    - The application is accessible from all network interfaces (host='0.0.0.0').

Usage:
    $ python run.py
"""

from flask import Flask
from flask_cors import CORS
from app.service import ai_model

app = Flask(__name__)
app.register_blueprint(ai_model)

CORS(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
