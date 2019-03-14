# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Facial Recognizer'
API_DESC = 'Recognize faces in an image and extract embedding vectors for each face'
API_VERSION = '1.1.0'

# default model
MODEL_NAME = API_TITLE
MODEL_ID = MODEL_NAME.lower().replace(' ', '-')
DEFAULT_MODEL_PATH = 'assets/facenet.pb'
MODEL_LICENSE = 'MIT'

DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 2
DEFAULT_PREPROCESS_THREADS = 2