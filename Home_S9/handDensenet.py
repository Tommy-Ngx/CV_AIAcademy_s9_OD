# to detect and recog hand using YOLO and MobileNet
# train based on SLT dataset
#
#
# edit: by Vu Hai ,
# date: 5 Oct 2019

#
#
import tensorflow as tf
import keras
import cv2 as cv
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import regularizers
import argparse
import sys
import numpy as np
import os.path
import darknet
import time
#import imutils
from keras.models import load_model, model_from_json
from keras.backend.tensorflow_backend import set_session

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "../model/obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# for hand detection
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "../model/yolo3_0_poses.cfg"
modelWeights = "../model/yolo3_0_poses_final.weights"
metafile ="../model/obj.data"

    
netMain = darknet.load_net_custom(modelConfiguration.encode(
            "ascii"), modelWeights.encode("ascii"), 0, 1)  # batch size = 1

metaMain = darknet.load_meta(metafile.encode("ascii"))

# for hand recognizer
#sys.path.insert(0, '../handSLT/mobilenet/')
#from mobilenet import MobileNetMod

#IMAGE_SIZE = 224  
def preProcImg(img_array):
  dim = (224, 224)
  
  img_array = cv.resize(img_array, dim, interpolation=cv.INTER_AREA)
  img_array = np.array(img_array).astype('float32')/255
  #np_image = np.array(np_image).astype('float32')/255
  #img_array = img_array/255
  #img_array = preprocess_input(img_array)
  img_expanded_dims = np.expand_dims(img_array, axis=0)
  
  return img_expanded_dims
  #return keras.applications.densenet.preprocess_input(img_expanded_dims)
  
#classMap = {0: 'hand', 1: 'ok', 2: 'paper', 3: 'rock',
#            4: 'scissors', 5: 'the-finger', 6: 'thumbdown', 7: 'thumbup'}
classMap = {0: 'ok', 1: 'paper', 2: 'rock',
            3: 'scissors', 4: 'the-finger', 5: 'thumbdown', 6: 'thumbup'}
#mnet = MobileNetMod(classMap)


# load json and create model
json_file = open('../model/DenseNetHandV2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelDenseNet = tf.keras.models.model_from_json(loaded_model_json)


#modelDenseNet=tf.keras.models.load_model('../model/DenseNetHandV2.h5')
modelDenseNet.load_weights('../model/DenseNetHandV2.h5')






def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
def write_log(x, y , w , h, classId,conf):
  logfile = "../video/video_SLT_11Oct.txt"
  filel = open(logfile, "a+")
  filel.write("{:4d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:2d},{:.2f}\n".format(int(fcount), 
               float(x), float(y), float(w), float(h), int(classId),float(conf)))
  filel.close()


def convertBack(x, y, w, h):
    left = int(round(x - (w / 2)))
    right = int(round(x + (w / 2)))
    top = int(round(y - (h / 2)))
    bottom = int(round(y + (h / 2)))
    return left, right, top, bottom
    
    
def postprocess(frame, outs,fcount):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    classname = 'hand'
    min_W=16
    min_H=32
    for detection in outs:
      if classname !=None and detection[0].decode()!=classname:
        continue
      x, y, w, h  = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
      left, right, top, bottom = convertBack(
            float(x), float(y), float(w), float(h))
      if top < 400:
        drawPredMobileNet(left, top, right, bottom)
      else:
        continue

def drawPredMobileNet(left, top, right, bottom):
    X = []
    origH = frame.shape[0]
    origW = frame.shape[1]
    newH = frame_resized.shape[0]
    newW = frame_resized.shape[1]
    
    ratioH = origH/newH + 0.01
    ratioW = origW/newW + 0.05
    
    if bottom > top and right > left:
        origTop = int(top*ratioH)-10
        origBottom = int(bottom*ratioH)+10
        origleft = int(left*ratioW)-10
        origright= int(right*ratioW)+10
        roi = frame[origTop:origBottom, origleft:origright,]
        
        #roi = frame_resized[top:bottom,left:right]
        #img_shape =(160, 120)
        if roi.shape[0] > 0 and roi.shape[1] > 0:  
          img = preProcImg(roi)
          predictions = modelDenseNet.predict(img)
          maxInRows = np.amax(predictions, axis=1)
          res = np.where(predictions == maxInRows)
          classId = int(res[1])
          label = classMap[classId]
        else:
          label = 'NG'
        if label == 'NG':
          return
        print(label)
        #print(maxInRows)
        cv.rectangle(frame_resized, (left, top), (right, bottom), (255, 178, 50), 3)
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame_resized, (left, top - round(1.5*labelSize[1])), (left + round(
            1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame_resized, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        #write_log(left,top,right-left,bottom-top,classId,maxInRows)
       
        
## main function from here
inputvideo = "../video/video10.mov"
outputFile = "../video/out_video10_denseNetV2.mp4"
cap = cv.VideoCapture(inputvideo)
#cap.set(3, 720)
#cap.set(4, 1280)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
# DIVX 
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15, (round(
        darknet.network_width(netMain)), round(darknet.network_width(netMain))))
print(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

# Create an image we reuse for each detect
darknet_image = darknet.make_image(darknet.network_width(netMain),
                                   darknet.network_height(netMain),3)
                                    
fcount = 0;
totalTime = 0;
totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
while (cap.isOpened()):
    # get frame from the video
    prev_time = time.time()
    hasFrame, frame = cap.read()
     
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break
    tick = time.process_time()
    # Create a 4D blob from a frame.
    #blob = cv.dnn.blobFromImage(
    #    frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    #net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    #outs = net.forward(getOutputsNames(net))
    #frame = imutils.rotate(frame,90)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    outs = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

    # Remove the bounding boxes with low confidence
    postprocess(frame_resized, outs, fcount)
    tock = time.process_time()
    proc_time = tock - tick
    #print(proc_time)
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #t, _ = darknet.getPerfProfile()
    #t = tock - tick
    label = "Processing time: %5.2f s" % (proc_time)
    cv.putText(frame_resized, label, (0, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    fcount = fcount+1
    # Write the frame with the detection boxes
    print(1.0/proc_time)
    frame_resized = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
    vid_writer.write(frame_resized.astype(np.uint8))
    totalTime = proc_time + totalTime
    print("processed frame %d -- %d with processed time %5.2f " % (fcount,totalFrames,totalTime))
    cv.imshow("result", frame_resized)
    #cv.waitKey(10)
# release video
vid_writer.release()
cap.release()
    #plt.axis("off")
    #plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    #plt.show()
    # cv.imshow(winName, frame)
