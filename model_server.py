from openvino import inference_engine as ie
from keras.applications import imagenet_utils
import numpy as np
import settings
import helpers
import redis
import json
import time

db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

def classify_process():

    net = ie.IENetwork.from_ir(model=settings.model_xml, weights=settings.model_bin)
    plugin = ie.IEPlugin(device="CPU")
    exec_net = plugin.load(network=net, num_requests=1)
    print("\t[INFO] Model loaded\n")

    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None


        for q in queue:
            # print("\t[INFO] Deserialize image and take out image\n")

            #deserialize the object and obtain input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"], settings.IMAGE_DTYPE, (1, settings.channel, settings.height, settings.width))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            # print("\t[INFO] Inferencing\n")
            print("\t[INFO] Batch size: {}\n".format(batch.shape))
            # print("\t[INFO] Batch size: {}".format(batch.shape))
            res = exec_net.infer(inputs={settings.input_blob: image})

            output_node_name = list(res.keys())[0]
            res = res[output_node_name]
            results = imagenet_utils.decode_predictions(res) #result

            # print("\t[INFO] Output predictions:\n")

            for (imageID, resultSet) in zip(imageIDs, results):
                output = [] #result print here
                for (imagenetID, label, prob) in results[0]:
                    r = {"label":label, "probability": float(prob)} #result label & probability
                    output.append(r)
                
                #store the output in Redis database using imageID as the key
                db.set(imageID, json.dumps(output))
                #remove classified images from queue
                # print("\t[INFO] Remove classified image\n")
                db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
            time.sleep(settings.SERVER_SLEEP)

if __name__=="__main__":
    classify_process()