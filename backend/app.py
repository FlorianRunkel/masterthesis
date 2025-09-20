import os
import sys
import logging

from flask import Flask
from flask_cors import CORS
from config import Config

from api_handlers.candidates.handler import candidates_bp
from api_handlers.linkedin.handler import linkedin_bp
from api_handlers.prediction.handler import prediction_bp
from api_handlers.user_management.handler import user_management_bp
from api_handlers.feedback.handler import feedback_bp

'''
Create and configure the backend application
'''
def create_app():

    app = Flask(__name__,
            template_folder=Config.TEMPLATE_DIR,
            static_folder=Config.STATIC_DIR)

    app.logger.setLevel(logging.INFO)

    '''
    CORS Origins
    '''
    origins = [
        "https://masterthesis-igbq.onrender.com",
        "https://masterthesis-backend.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:0",
        "https://*.amazonaws.com",
        "https://*.elasticbeanstalk.com",
        "https://*.ecs.amazonaws.com",
    ]

    CORS(app,
         origins=origins,
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-User-Uid"],
         supports_credentials=True)

    app.register_blueprint(candidates_bp)
    app.register_blueprint(linkedin_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(user_management_bp)
    app.register_blueprint(feedback_bp)

    @app.route('/')
    def root():
        return {'status': 'ok', 'service': 'masterthesis-backend'}, 200

    app.logger.info("Flask application successfully created and configured.")

    return app

'''
Initialize and start the Flask application
'''
app = create_app()

'''
Run the application when executed directly
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)