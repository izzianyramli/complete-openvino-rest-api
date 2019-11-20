import requests
import cv2
from datetime import datetime

URL = "http://127.0.0.1:5000/predict"
IMAGE_PATH = "elephant.jpg"

print("\t[INFO] Image file name: [",IMAGE_PATH,"]\n")
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

image = cv2.imread(IMAGE_PATH)
cv2.imshow("image", image)
cv2.waitKey(3000)
cv2.destroyAllWindows()

start_time = datetime.now().time().strftime('%H:%M:%S')
# print ("\t[INFO] image type from client = {}, class:{}\n".format(image.dtype, type(image)))
# print ("\t[INFO] Image size = {}, shape = {}, mean intensity = {}\n".format(image.size, image.shape, image.mean()))
print("\t[INFO] Loading image to server\n")
r = requests.post(URL, files=payload).json()
print("\t[INFO] Image passed to server\n")

if r["success"]:
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
else:
	print("Request failed")

end_time = datetime.now().time().strftime('%H:%M:%S')
total_time=(datetime.strptime(end_time,'%H:%M:%S') - datetime.strptime(start_time,'%H:%M:%S'))
print ("Time interval:{} \n".format(total_time))
