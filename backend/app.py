from flask import Flask
from flask_cors import CORS
from config import Config
from db import init_db
from routes.upload import upload_bp
from routes.analyze import analyze_bp
from routes.chat import chat_bp
from routes.history import history_bp
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    CORS(app)
    init_db()

    # register blueprints
    app.register_blueprint(upload_bp, url_prefix="/api")
    app.register_blueprint(analyze_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")

    @app.route("/api/health")
    def health():
        return {"status": "ok"}, 200

    return app

if __name__ == "__main__":
    create_app().run(debug=True, host="0.0.0.0", port=8000)
