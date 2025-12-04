from flask import Flask
from flask_cors import CORS
from api.routes import api_bp

app = Flask(__name__)
CORS(app)

# Register the API Blueprints
app.register_blueprint(api_bp, url_prefix='/api/v1')

# if __name__ == '__main__':
#     # Threaded=True handles multiple requests better
#     app.run(port=5000, debug=False, threaded=True, use_reloader=False)
    # app.run(debug=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

# api key open router
# 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