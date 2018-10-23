import requests
import pytest


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Server'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'facenet-tensorflow'
    assert metadata['name'] == 'facenet TensorFlow Model'
    assert metadata['description'] == 'facenet TensorFlow model trained on LFW data to detect faces and generate '\
                                      'embeddings'
    assert metadata['license'] == 'MIT'


def test_predict():

    model_endpoint = 'http://localhost:5000/model/predict'

    # Test by the image with multiple faces
    img1_path = 'assets/codait.jpeg'

    with open(img1_path, 'rb') as file:
        file_form = {'image': (img1_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) > 0
    assert len(response['predictions'][0]['detection_box']) == 4
    assert len(response['predictions'][0]['embedding']) == 512

    # Test by the image without faces
    img2_path = 'assets/IBM.jpeg'

    with open(img2_path, 'rb') as file:
        file_form = {'image': (img2_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'
    assert len(response['predictions']) == 0

    # Test by the text data
    img3_path = 'assets/README.md'

    with open(img3_path, 'rb') as file:
        file_form = {'image': (img3_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__])
