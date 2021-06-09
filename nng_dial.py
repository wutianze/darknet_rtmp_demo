from pynng import Pair0
import cv2
import struct
import time
image = cv2.imread("./data/dog.jpg")
height, width, channel = image.shape
image_bytes = image.tobytes()
bytes_len = len(image_bytes)
data = struct.pack('iii'+str(bytes_len)+'s',height,width,channel,image_bytes)
'''
image_array = numpy.frombuffer(image_bytes,dtype=numpy.uint8)
print(image_array.shape)
image_array2 = image_array.reshape(width,height,channel)
print(image_array2.shape)
img_for_detect = darknet.make_image(width, height, channel)
data = struct.pack('iii'+str(bytes_len)+'s',width,height,channel,image_bytes)
darknet.copy_image_from_bytes(img_for_detect,data)
'''
with Pair0(dial="tcp://127.0.0.1:13131") as s0:
    while True:
        s0.send(data)
        time.sleep(0.005)
