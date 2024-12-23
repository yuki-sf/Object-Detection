from Detector import *
import os

def main():
    type = str(input("Enter the path of the video :  "))
    videoPath = type

    #  give path for video render "test-videos/video2.mp4"
    # 0 for webcam

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == "__main__":
    main()