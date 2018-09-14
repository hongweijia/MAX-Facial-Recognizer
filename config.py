# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'Model Asset Exchange Server'
API_DESC = 'An API for serving models'
API_VERSION = '0.1'

# default model
MODEL_NAME = 'facenet'
DEFAULT_MODEL_PATH = 'assets/{}.pb'.format(MODEL_NAME)
MODEL_LICENSE = 'MIT'

MODEL_META_DATA = {
    'id': '{}-tensorflow'.format(MODEL_NAME.lower()),
    'name': '{} TensorFlow Model'.format(MODEL_NAME),
    'description': '{} TensorFlow model trained on LFW data to detect faces and generate embeddings'.format(MODEL_NAME),
    'type': 'face_recognition',
    'license': '{}'.format(MODEL_LICENSE)
}

DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 2
DEFAULT_PREPROCESS_THREADS = 2