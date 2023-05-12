from flask import Flask

app = Flask(__name__)

@app.route("/")
def start_application():
    return "It works . . ."

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
