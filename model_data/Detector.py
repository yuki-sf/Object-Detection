import cv2
import numpy as np
import time

np.random.seed(15)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        # self.classes = None
        # self.confThreshold = 0.5
        # self.nmsThreshold = 0.4

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')

        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        # print(self.classesList)

    def onVideo(self):
        print("Starting video capture...")
        if self.videoPath == '0':
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # to prevent select() timout issues
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # to reduce lag by reducing resolution 
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # change contrast, brightness, etc using --> sudo guvcview
        else:
            videoName = 'test-videos/'+self.videoPath+'.mp4'  # reducing the path and fileformat    
            cap = cv2.VideoCapture(videoName)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return
        
        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2) # nms_threshold --> >20% overlap will be displayed
            
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence) # show confidence for 2 decimal places

                    x, y, w, h = bbox  # x, y, width, height

                    cv2.rectangle(image, (x, y), (x+w, y+h), color = classColor, thickness = 1)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                    # draw CORNER THICK lines around bounding box

                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    cv2.line(image,(x,y), (x + lineWidth, y), classColor, thickness = 3)
                    cv2.line(image,(x,y), (x, y + lineWidth), classColor, thickness = 3)
                    
                    cv2.line(image,(x + w, y), (x + w - lineWidth, y), classColor, thickness = 3)
                    cv2.line(image,(x + w, y), (x + w, y + lineWidth), classColor, thickness = 3)

                    # 

                    cv2.line(image,(x,y + h), (x + lineWidth, y + h), classColor, thickness = 3)
                    cv2.line(image,(x,y + h), (x, y + h - lineWidth), classColor, thickness = 3)
                    
                    cv2.line(image,(x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness = 3)
                    cv2.line(image,(x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness = 3)

            cv2.putText(image, "FPS: " + str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,200,0), 2)
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

            (success, image) =cap.read()
        cv2.destroyAllWindows()