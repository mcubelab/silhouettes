import datetime
import time
import os

while True:
   t = str(datetime.datetime.now())
   directory = "/home/mcube/shapes_weights/" + t + "/"
   if not os.path.exists(directory):
       os.makedirs(directory)
   os.system("scp -i ~/ssh_keys/shapes_aws_key2.pem -r ubuntu@ec2-34-224-79-79.compute-1.amazonaws.com:/home/ubuntu/weights/ " + directory)
   time.sleep(5*60) # We wait 10 min
