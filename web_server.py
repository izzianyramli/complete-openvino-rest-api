from openvino import inference_engine as ie
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np 
import settings
import helpers
import cv2
import flask
import json
import io
import os
import uuid
import sys
import redis
import base64
import time

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

def process_image(image, target):
    # image.show()

    if image.mode != "RGB":
        image = image.mode("RGB")

    image = image.resize(target)
    image = (np.array(image) - 0) / 255.0
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, settings.channel, settings.height, settings.width)

    # print("\t[INFO] Image pre-processing done")

    return image

@app.route("/")
def home():
    return ("Hi!")

#server REST API with /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # print("\t[INFO] Receiving image\n")
            image = flask.request.files["image"].read()

            # print("\t[INFO] Reading image")
            image = Image.open(io.BytesIO(image))

            # print("\t[INFO] Pre-processing image\n")
            image = process_image(image, (settings.width, settings.height))

            image = image.copy(order="C")

            # print("\t[INFO] Saving image into Redis\n")
            k = str(uuid.uuid4())
            d = {"id": k, "image": helpers.base64_encode_image(image)}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
            # print("\t[INFO] Image saved into Redis\n")

            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)

            data["success"] = True

        else:
            flask.jsonify({'error': 'no file'}), 400

    return flask.jsonify(data)

if __name__ == "__main__":
    print("\t[INFO] Starting web service\n")
    app.run()