import base64
import sys
import numpy as np

#serialize image to store in Redis
def base64_encode_image(a):
    # print("\t[INFO] Encoding image")
    return base64.b64encode(a).decode("utf-8")

#deserialize image to pass through model
def base64_decode_image(a, dtype, shape):
    # print("\t[INFO] Decoding image")

    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a