import numpy as np
import cv2
CONFIDENCE = 0.01
THRESHOLD = 0.3
import time
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image
import string
import numpy as np
import cv2

import predicthelper as predictmodule





reader = easyocr.Reader(['en'])
ALLOWED_LIST = string.ascii_uppercase+string.digits


# -----------  load the trained model  -----------


#here we are loading yolov3 weights

plate_net = cv2.dnn.readNetFromDarknet(r'./models/plate.cfg', r'./models/plate.weights')
plate_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
plate_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)




#This function takes image and then apply model to extract plate reion from car images
def classify_plate(image):
        c=0
        idxs = []
        layerOutputs =[]
        try:
                (H, W) = image.shape[:2]
                ln = plate_net.getLayerNames()
                ln = [ln[i[0] - 1] for i in plate_net.getUnconnectedOutLayers()]
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
                plate_net.setInput(blob)
                layerOutputs = plate_net.forward(ln)
        except Exception as e:
                print("PLATE EXTRACTION ERROR ", e)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
                for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        if confidence > CONFIDENCE:
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        print('cropaya')
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        print(x,y,w,h)
                        x-=5
                        if x<0:
                                x = 0
                        if  y <0:
                                y = 0
                        w+=10
                        #here we get the bounding box of number plate
                        crop_img = image[y:y+h, x:x+w]
                        cv2.imshow(str(c),crop_img)
                        #we saved the extracted plate on disk for prediction
                        cv2.imwrite("./outputimage/"+str(c)+".png",crop_img)
                        c =c+1
                        return c
        return -1
        print('nhicropaya')


def classify_plate_in_video(cap):
  frame_rate = 10
  prev = 0
  c=0
  cnt=0
  while (cap.isOpened()): 
    ret, frame = cap.read()
    image  =frame
    cnt+=1
    if ret == True:
                        idxs = []
                        layerOutputs =[]
                        try:
                                (H, W) = image.shape[:2]
                                ln = plate_net.getLayerNames()
                                ln = [ln[i[0] - 1] for i in plate_net.getUnconnectedOutLayers()]
                                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
                                plate_net.setInput(blob)
                                layerOutputs = plate_net.forward(ln)
                        except Exception as e:
                                print("PLATE EXTRACTION ERROR ", e)
                        
                        boxes = []
                        confidences = []
                        classIDs = []
                        for output in layerOutputs:
                                for detection in output:
                                        scores = detection[5:]
                                        classID = np.argmax(scores)
                                        confidence = scores[classID]
                                        if confidence > CONFIDENCE:
                                                box = detection[0:4] * np.array([W, H, W, H])
                                                (centerX, centerY, width, height) = box.astype("int")
                                                x = int(centerX - (width / 2))
                                                y = int(centerY - (height / 2))
                                                boxes.append([x, y, int(width), int(height)])
                                                confidences.append(float(confidence))
                                                classIDs.append(classID)
                                                idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
                        if len(idxs) > 0:
                                # loop over the indexes we are keeping
                                print('cropaya')
                                for i in idxs.flatten():
                                        (x, y) = (boxes[i][0], boxes[i][1])
                                        (w, h) = (boxes[i][2], boxes[i][3])
                                        if x<0 or y<0 or w<0 or h<0:
                                                break
                                        crop_img = image[y:y+h, x:x+w]
                                        if(cnt%10==0):
                                            cv2.imwrite("./outputimage/"+str(c)+".png",crop_img)
                                        c=c+1
                                        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
  
  cap.release()
  return c



c = 0
## To Run on image
#image = cv2.imread('32.jpg')
#cv2.imshow('Image',image)
#c =classify_plate(image)

## To Run on Video
#cap = cv2.VideoCapture('2.mp4')
#c = classify_plate_in_video(cap)

#10,109,113,t2
#10
#2 one mis
#one one extra noise
#inr one identified 6 char 16
#18 not detected
#21 one missing and one misclas

img=cv2.imread("./final_test/1.jpg")

row, col = img.shape[:2]
bottom = img[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

bordersize = 30
border = cv2.copyMakeBorder(
    img,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean,mean,mean]
)

c=classify_plate(border)

if(c==-1):
    print("numberplate not found")
else:
    last = "0000"
    k =1
    print(c)

    for i in range(c):
        #loading image which was previously saved on the disk
        image = cv2.imread("./outputimage/"+str(i)+".png")
        #now give the extracted image to predcit function which use preprocessing and easyocr for prediction
        prediction = predictmodule.predict(image,reader,ALLOWED_LIST)
        if prediction[-4:]!=last:
            print("Car" + str(k)+ "plate number is " + prediction)
            k = k+1
            last  = prediction[-4:]



#19 -ok
#18 two mis G with 6 and M with H
#17 one mis M with H and one 1 at very corner is missing this is because yolo has given very small bounding box
#16-extremem dark yet ok
#15- too much noise not able to segment
#14 -missnig 4 char because of too noise
#13- one mis class
#12-ok
#11-ok
#10-ok
#9-ok
#8-ok
#7-ok
#6-ok
#5-ok
#4-ok
#3-one mis
#2-one extra but all classified
#1-ok

