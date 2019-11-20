from threading import Thread
import numpy as np
import requests
import time
from datetime import datetime

URL = "http://localhost:5000/predict"
IMAGE_PATH = "elephant.jpg"

NUM_REQUESTS = 50
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
    
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}

	start_time = datetime.now().time().strftime('%H:%M:%S')
	# print("Start time:  ", start_time)

	r = requests.post(URL, files=payload).json()

	if r["success"]:
		print ("[INFO] thread {} OK".format(n))
		
	else:
		print ("[INFO] thread {} FAILED".format(n))
	
	end_time = datetime.now().time().strftime('%H:%M:%S')
	# print("End time:  ", end_time)

	total_time=(datetime.strptime(end_time,'%H:%M:%S') - datetime.strptime(start_time,'%H:%M:%S'))
	print ("Time interval:{} \n".format(total_time))

# loop over the number of threads
for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	t = Thread(target=call_predict_endpoint, args=(i,))
	t.daemon=True

	t.start()
	time.sleep(SLEEP_COUNT)
	t.join() #to make sure next thread start after previous thread finish.
	# print("\tDone!\n")

time.sleep(30)