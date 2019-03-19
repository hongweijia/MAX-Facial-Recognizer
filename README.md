[![Build Status](https://travis-ci.com/IBM/MAX-Facial-Recognizer.svg?branch=master)](https://travis-ci.com/IBM/MAX-Facial-Recognizer) [![Website Status](https://img.shields.io/website/http/max-facial-recognizer.max.us-south.containers.appdomain.cloud/swagger.json.svg?label=api+demo)](http://max-facial-recognizer.max.us-south.containers.appdomain.cloud/)

# IBM Code Model Asset Exchange: Facial Recognizer

This repository contains code to instantiate and deploy a face detection and feature extraction model. The model first detects faces in an input image and then generates an embedding vector for each face. The generated embeddings can be used for downstream tasks such as classification, clustering, verification etc. The model accepts an image as input and returns the bounding box coordinates, probability and embedding vector for each face detected in the image.

The model is based on the [FaceNet model](https://github.com/davidsandberg/facenet). The model files are hosted on [IBM Cloud Object Storage](http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/facenet.tar.gz). The code in this repository deploys the model as a web service in a Docker container. This repository was developed
as part of the [IBM Code Model Asset Exchange](https://developer.ibm.com/code/exchanges/models/).

## Model Metadata
| Domain | Application | Industry  | Framework | Training Data | Input Data Format |
| ------------- | --------  | -------- | --------- | --------- | -------------- | 
| Vision | Face Detection | General | TensorFlow | [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) | Image (RGB) | 

## References

* _Florian Schroff, Dmitry Kalenichenko, James Philbin_, ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832), CVPR 2015.
* _Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman_, ["VGGFace2: A dataset for recognising face across pose and age"](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf), International Conference on Automatic Face and Gesture Recognition, 2018.
* [FaceNet Github Repository](https://github.com/davidsandberg/facenet).

## Licenses

| Component | License | Link  |
| ------------- | --------  | -------- |
| This repository | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [LICENSE](LICENSE) |
| Model Weights | [MIT](https://opensource.org/licenses/MIT) | [LICENSE](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) |
| Model Code (3rd party) | [MIT](https://opensource.org/licenses/MIT) | [LICENSE](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md) |
| Test assets | Various | [Asset README](assets/README.md) |

## Pre-requisites:

* `docker`: The [Docker](https://www.docker.com/) command-line interface. Follow the [installation instructions](https://docs.docker.com/install/) for your system.
* The minimum recommended resources for this model is 2GB Memory and 1 CPUs.

# Steps

1. [Deploy from Docker Hub](#deploy-from-docker-hub)
2. [Deploy on Kubernetes](#deploy-on-kubernetes)
3. [Run Locally](#run-locally)

## Deploy from Docker Hub

To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 codait/max-facial-recognizer
```

This will pull a pre-built image from Docker Hub (or use an existing image if already cached locally) and run it.
If you'd rather checkout and build the model locally you can follow the [run locally](#run-locally) steps below.

## Deploy on Kubernetes

You can also deploy the model on Kubernetes using the latest docker image on Docker Hub.

On your Kubernetes cluster, run the following commands:

```
$ kubectl apply -f https://github.com/IBM/MAX-Facial-Recognizer/raw/master/max-facial-recognizer.yaml
```

The model will be available internally at port `5000`, but can also be accessed externally through the `NodePort`.

## Run Locally

1. [Build the Model](#1-build-the-model)
2. [Deploy the Model](#2-deploy-the-model)
3. [Use the Model](#3-use-the-model)
4. [Run the Notebook](#4-run-the-notebook)
5. [Development](#5-development)
6. [Clean Up](#6-cleanup)


### 1. Build the Model

Clone this repository locally. In a terminal, run the following command:

```
$ git clone https://github.com/IBM/MAX-Facial-Recognizer.git
```

Change directory into the repository base folder:

```
$ cd MAX-Facial-Recognizer
```

To build the docker image locally, run: 

```
$ docker build -t max-facial-recognizer .
```

All required model assets will be downloaded during the build process. _Note_ that currently this docker image is CPU only (we will add support for GPU images later).


### 2. Deploy the Model

To run the docker image, which automatically starts the model serving API, run:

```
$ docker run -it -p 5000:5000 max-facial-recognizer
```

### 3. Use the Model

The API server automatically generates an interactive Swagger documentation page. Go to `http://localhost:5000` to load it. From there you can explore the API and also create test requests.
Use the `model/predict` endpoint to load a test image (you can use one of the test images from the `assets` folder) and get predicted labels for the image from the API.

![Swagger UI Screenshot](docs/swagger-screenshot.png)

You can also test it on the command line, for example:

```
$ curl -F "image=@assets/Lenna.jpg" -XPOST http://localhost:5000/model/predict
```

You should see a JSON response like that below:

```
{
  "status": "ok",
  "predictions": [
    {
      "detection_box": [
        85.5373387336731,
        77.938033670187,
        149.2407527267933,
        170.62581571377814
      ],
      "probability": 0.9959015250205994,
      "embedding": [
        -0.016315622255206108,
        -0.04482162743806839,
        -0.02662980556488037,
        -0.003268358064815402,
        0.0253919567912817,
        0.07166660577058792,
        -0.0225024726241827,
        ...
        -0.05647726729512215
      ]
    }
  ]
}
```
### 4. Run the Notebook

Once the model server is running, you can see how to use it by walking through [the demo notebook](demo.ipynb). _Note_ the demo requires `jupyter`, `numpy`, `scipy`, `matplotlib`, `Pillow`, and `requests`.

Run the following command from the model repo base folder, in a new terminal window (leaving the model server running in the other terminal window):

```
$ jupyter notebook
```

This will start the notebook server. You can open the simple demo notebook by clicking on `demo.ipynb`. There is also a second demo illustrating a privacy use-case, which you can open by clicking on `demo_gdpr.ipynb`.

### 5. Development

To run the Flask API app in debug mode, edit `config.py` to set `DEBUG = True` under the application settings. You will then need to rebuild the docker image (see [step 1](#1-build-the-model)).

### 6. Cleanup

To stop the Docker container, type `CTRL` + `C` in your terminal.
