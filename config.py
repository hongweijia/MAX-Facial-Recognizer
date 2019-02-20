# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Facial Recognizer'
API_DESC = 'Recognize faces in an image and extract embedding vectors for each face'
API_VERSION = '0.1'

# default model
MODEL_NAME = 'MAX Facial Recognizer'
DEFAULT_MODEL_PATH = 'assets/facenet.pb'
MODEL_LICENSE = 'MIT'

DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 2
DEFAULT_PREPROCESS_THREADS = 2