from waitress import serve
from ahp_app import app  # 위에서 만든 Flask app

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=8000)
