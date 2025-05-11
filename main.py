import cv2 as cv 
import numpy as np
import pyttsx3
import time
import serial
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)

path = r"ReferenceImages\dog.png"
engine = pyttsx3.init('sapi5')           
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolo.weights', 'yolo.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" %(class_names[classid[0]], score)
        la = class_names[classid[0]]
        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        
        if la =='dog': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("dog HERE")
            speak("dog Here")
            arduino.write(b'A')
            time.sleep(2)
        elif la =='cat': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("cat HERE")
            speak("cat Here")
            arduino.write(b'B')
            time.sleep(2)
        elif la =='cow': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("cow HERE")
            speak("cow Here")
            arduino.write(b'C')
            time.sleep(2)
        elif la =='elephant': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("elephant HERE")
            speak("elephant Here")
            arduino.write(b'D')
            time.sleep(2)
        elif la =='horse': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("horse HERE")
            speak("horse Here")
            arduino.write(b'E')
            time.sleep(2)
        elif la =='girafe': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("girafe HERE")
            speak("girafe Here")
            arduino.write(b'F')
            time.sleep(2)
        elif la =='sheep': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("sheep HERE")
            speak("sheep Here")
            arduino.write(b'G')
            time.sleep(2)
        elif la =='zebra': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("zebra HERE")
            speak("zebra Here")
            arduino.write(b'H')
            time.sleep(2)
        elif la =='bear': 
            cv.putText(image, la, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            print("bear HERE")
            speak("bear Here")
            arduino.write(b'I')
            time.sleep(2)
        else:
            print("Waiting for data")    
        time.sleep(1)   
        cv.imshow('Animal Detection Application',frame)    
      
        if classid ==15: # cat
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==16: # dog
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==17: # horse
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==18: # sheep
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==19: # cow
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==20: # elephant
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==21: # bear
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==22: # zebra
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==23: # girafe
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
       
        
    return data_list


# cap = cv.VideoCapture(path)
cap = cv.VideoCapture(0)
counts = 0
while True:
    ret, frame = cap.read()
    imgS = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    data = object_detector(frame)
    # cv.waitkey(0)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

