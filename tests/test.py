import requests
import pytest


def test_response():

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