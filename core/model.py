import io
import logging
import math

from flask_restplus import abort
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.ops import data_flow_ops
from maxfw.model import MAXModelWrapper

import core.src.facenet as facenet
from config import DEFAULT_MODEL_PATH, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE, \
    DEFAULT_PREPROCESS_THREADS, MODEL_NAME
from core.src.align.align_dataset_mtcnn import load_mtcnn, run_mtcnn

logger = logging.getLogger()

class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': '{}-tensorflow'.format(MODEL_NAME.lower()),
        'name': '{} TensorFlow Model'.format(MODEL_NAME),
        'description': '{} TensorFlow model trained on LFW data to detect faces '
                       'and generate embeddings'.format(MODEL_NAME),
        'type': 'face_recognition',
        'license': 'MIT',
        'source': 'https://developer.ibm.com/exchanges/models/all/max-facial-recognizer/'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        self.graph_mtcnn = tf.Graph()
        with self.graph_mtcnn.as_default():
            self.sess_mtcnn = tf.Session(graph=self.graph_mtcnn)
            self.pnet, self.rnet, self.onet = load_mtcnn(self.sess_mtcnn)

        self.graph_embedding = tf.Graph()
        with self.graph_embedding.as_default():
            self.sess_embedding = tf.Session(graph=self.graph_embedding)
            self.model_path = path
            self.load_embedding_model(self.model_path)

        self.warmup()

        logger.info('Finished loading and warming up model')

    def warmup(self):
        sample = np.random.randint(0, 255,
                                   [1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3],
                                   dtype=np.uint8)
        self.gen_embedding(sample)

    def detect_faces(self, image):
        bboxes, faces = run_mtcnn(image, self.pnet, self.rnet, self.onet,
                                  image_size=DEFAULT_IMAGE_SIZE)
        return bboxes, faces

    def create_input_pipeline(self, input_queue, image_size,
                              nrof_preprocess_threads, batch_size_placeholder):
        images_and_labels_list = []
        for _ in range(nrof_preprocess_threads):
            images_input, label, control = input_queue.dequeue()
            images = []
            for image in [images_input]:
                image = tf.cond(facenet.get_control_flag(control[0], facenet.RANDOM_ROTATE),
                                lambda: tf.py_func(facenet.random_rotate_image, [image], tf.uint8),
                                lambda: tf.identity(image))
                image = tf.cond(facenet.get_control_flag(control[0], facenet.RANDOM_CROP),
                                lambda: tf.random_crop(image, image_size + (3,)),
                                lambda: tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
                image = tf.cond(facenet.get_control_flag(control[0], facenet.RANDOM_FLIP),
                                lambda: tf.image.random_flip_left_right(image),
                                lambda: tf.identity(image))
                image = tf.cond(facenet.get_control_flag(control[0], facenet.FIXED_STANDARDIZATION),
                                lambda: (tf.cast(image, tf.float32) - 127.5) / 128.0,
                                lambda: tf.image.per_image_standardization(image))
                image = tf.cond(facenet.get_control_flag(control[0], facenet.FLIP),
                                lambda: tf.image.flip_left_right(image),
                                lambda: tf.identity(image))
                image.set_shape(image_size + (3,))
                images.append(image)
            images_and_labels_list.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels_list, batch_size=batch_size_placeholder,
            shapes=[image_size + (3,), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * 100,
            allow_smaller_final_batch=True)

        return image_batch, label_batch

    def load_embedding_model_help(self, model_path, nrof_preprocess_threads=1):
        image_placeholder = tf.placeholder(tf.uint8,
                                           (None, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
                                           name="input_images")
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_size = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                                   dtypes=[tf.uint8, tf.int32, tf.int32],
                                                   shapes=[(DEFAULT_IMAGE_SIZE,
                                                            DEFAULT_IMAGE_SIZE, 3), (1,), (1,)],
                                                   shared_name=None, name=None)

        eval_enqueue_op = eval_input_queue.enqueue_many([image_placeholder,
                                                         labels_placeholder,
                                                         control_placeholder],
                                                        name='eval_enqueue_op')
        image_batch, label_batch = self.create_input_pipeline(eval_input_queue,
                                                              image_size,
                                                              nrof_preprocess_threads,
                                                              batch_size_placeholder)

        # Load the model
        input_map = {'image_batch': image_batch, 'label_batch': label_batch,
                     'phase_train': phase_train_placeholder}
        facenet.load_model(model_path, input_map=input_map)

        # Get output tensor
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        return eval_enqueue_op, embeddings, labels_placeholder, \
               control_placeholder, phase_train_placeholder, \
               batch_size_placeholder, label_batch, image_placeholder

    def load_embedding_model(self, model_path):
        self.eval_enqueue_op, self.embeddings, self.labels_placeholder, \
        self.control_placeholder, self.phase_train_placeholder, \
        self.batch_size_placeholder, self.label_batch, self.image_placeholder = \
            self.load_embedding_model_help(model_path,
                                           nrof_preprocess_threads=DEFAULT_PREPROCESS_THREADS)

    def gen_embedding(self, faces):
        total_files = faces.shape[0]
        logger.info("Number of detected faces: {}".format(total_files))

        labels_array = np.expand_dims(np.arange(0, total_files), 1)
        control_array = np.zeros_like(labels_array, np.int32)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=self.sess_embedding)

        feed_dict = {self.image_placeholder: faces,
                     self.labels_placeholder: labels_array,
                     self.control_placeholder: control_array}
        self.sess_embedding.run(self.eval_enqueue_op, feed_dict)

        batch_size = math.gcd(DEFAULT_BATCH_SIZE, total_files)
        num_batches = total_files // batch_size

        all_embeddings = []
        all_labels = []
        for _ in range(num_batches):
            feed_dict = {self.phase_train_placeholder: False,
                         self.batch_size_placeholder: batch_size}
            embs, labels = \
                self.sess_embedding.run([self.embeddings, self.label_batch],
                                         feed_dict=feed_dict)
            all_embeddings.extend(embs)
            all_labels.extend(labels)
        return all_embeddings, all_labels

    def _read_image(self, image_data):
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        return image

    def _pre_process(self, x):
        try:
            image = self._read_image(x)
            return image
        except IOError as e:
            abort(400, "Please submit a valid image in PNG, or Tiff format")

    def _post_process(self, preds):
        # Modify this code if the schema is changed
        label_preds = [{'detection_box': p[0][0:4], 'probability': p[0][4],
                        'embedding': p[1].tolist()} for p in preds]
        return label_preds

    def _predict(self, x):
        # Detect the faces and bounding box from the input image
        bboxes, faces = self.detect_faces(x)

        if (len(faces) > 0):
            # Generate the embeddings for each detected faces
            faces = np.stack(faces)
            all_embeddings, all_labels = self.gen_embedding(faces)

            # Match the bounding boxes and embeddings
            results = [(bboxes[label], emb) for (label, emb) in
                       zip(all_labels, all_embeddings)]
            return results
        else:
            # for the cases without detected faces
            return []
