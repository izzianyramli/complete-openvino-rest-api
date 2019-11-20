import os
from openvino import inference_engine as ie


IMAGE_QUEUE = "image_queue"
CLIENT_SLEEP = 0.25
SERVER_SLEEP = 0.25
BATCH_SIZE = 32
IMAGE_DTYPE = "float64"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

model_xml = "C:/Users/nurulizz/Documents/MSC/Co/keras_openvino/model/frozen_model.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin"

net = ie.IENetwork.from_ir(model=model_xml, weights=model_bin)

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

batch, channel, height, width = net.inputs[input_blob].shape