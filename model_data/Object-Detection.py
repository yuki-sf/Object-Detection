import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import time


class ObjectDetector:
    def __init__(self, configPath, modelPath, classesPath):
        self.net = cv2.dnn_DetectionModel(modelPath, configPath)
        self.net.setInputSize(320, 320)  # Default size for faster processing
        self.net.setInputScale(1.0 / 255.0)
        self.net.setInputMean((0, 0, 0))
        self.net.setInputSwapRB(True)

        # Load class names
        with open(classesPath, 'r') as f:
            self.classesList = f.read().strip().splitlines()

        # Generate random colors for each class
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def detect(self, frame):
        classIDs, confidences, boxes = self.net.detect(frame, confThreshold=0.5)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Non-Maximum Suppression
        if len(indices) > 0:
            classIDs = [classIDs[i] for i in indices.flatten()]
            confidences = [confidences[i] for i in indices.flatten()]
            boxes = [boxes[i] for i in indices.flatten()]
        return classIDs, confidences, boxes

    def drawDetections(self, frame, classIDs, confidences, boxes):
        for classID, confidence, box in zip(classIDs, confidences, boxes):
            color = [int(c) for c in self.colorList[classID]]
            label = f"{self.classesList[classID]}: {confidence:.2f}"
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("400x300")

        self.model_choice = tk.StringVar(value="yolov4-tiny")
        self.video_source = None

        # UI Elements
        tk.Label(root, text="Object Detection", font=("Arial", 16)).pack(pady=10)

        # Model selection
        tk.Label(root, text="Select YOLO Version:").pack(anchor="w", padx=20)
        tk.Radiobutton(root, text="YOLOv4-Tiny (Light)", variable=self.model_choice, value="yolov4-tiny").pack(anchor="w", padx=40)
        tk.Radiobutton(root, text="YOLOv4 (Normal)", variable=self.model_choice, value="yolov4").pack(anchor="w", padx=40)

        # Input selection
        tk.Button(root, text="Select Video File", command=self.select_video).pack(pady=10)
        tk.Button(root, text="Use Camera", command=self.use_camera).pack(pady=10)

        # Start detection
        tk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=20)

    def select_video(self):
        self.video_source = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if self.video_source:
            messagebox.showinfo("Video Selected", f"Selected: {self.video_source}")

    def use_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows compatibility
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera could not be opened. Check permissions or device connection.")
            return
        self.video_source = 0  # Set to camera source
        messagebox.showinfo("Camera Selected", "Using live camera as input.")
        cap.release()

    def start_detection(self):
        # Validate input
        if self.video_source is None:
            messagebox.showerror("Error", "Please select a video file or camera.")
            return

        # Choose the YOLO model
        if self.model_choice.get() == "yolov4-tiny":
            configPath = "yolov4-tiny.cfg"
            modelPath = "yolov4-tiny.weights"
        else:
            configPath = "yolov4.cfg"
            modelPath = "yolov4.weights"

        classesPath = "coco.names"
        detector = ObjectDetector(configPath, modelPath, classesPath)

        # Open the video source
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video source.")
            return

        prev_frame_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detection and drawing
            classIDs, confidences, boxes = detector.detect(frame)
            frame = detector.drawDetections(frame, classIDs, confidences, boxes)

            # Calculate FPS
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time != 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Object Detection", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
