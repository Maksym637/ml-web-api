from flask import Flask
from app.service import ai_model
from flask_cors import CORS

app = Flask(__name__)
app.register_blueprint(ai_model)

CORS(app)

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
