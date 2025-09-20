import os

"""
AWS Config for the backend
This file contains the configuration for the backend, which is specific to the AWS environment.
"""
class AWSConfig:
    '''
    AWS Region
    '''
    AWS_REGION = os.getenv('AWS_REGION', 'eu-central-1')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

    '''
    Service Name and Version
    '''
    SERVICE_NAME = 'masterthesis-backend'
    SERVICE_VERSION = '1.0.0'

    '''
    Port and Host Configuration
    '''
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')

    '''
    ML-Modelle Path
    '''
    ML_MODELS_DIR = os.getenv('ML_MODELS_DIR', '/app/ml_pipe/models')

    '''
    MongoDB Configuration
    '''
    MONGODB_URI = os.getenv('MONGODB_URI')
    MONGODB_DB = os.getenv('MONGODB_DB', 'masterthesis')

    '''
    Logging Configuration
    '''
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    '''
    CORS Origins
    '''
    ALLOWED_ORIGINS = [
        "https://masterthesis-igbq.onrender.com",  # Frontend
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # AWS URLs (werden später hinzugefügt)
        "https://*.amazonaws.com",
        "https://*.elasticbeanstalk.com",
        "https://*.ecs.amazonaws.com",
    ]
