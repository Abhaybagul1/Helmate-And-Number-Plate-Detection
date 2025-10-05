import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import main

def process_image(image_path):
    img = cv2.imread(image_path)
    cv2.imshow("Selected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        threading.Thread(target=process_image, args=(file_path,)).start()

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        threading.Thread(target=process_video, args=(file_path,)).start()

def start_webcam_thread():
    threading.Thread(target=start_webcam).start()

# UI Setup
root = tk.Tk()
root.title("Helmet & Number Plate Detection")
root.geometry("400x300")

title_label = tk.Label(root, text="Helmet & Number Plate Detection", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20)
upload_image_btn.pack(pady=5)

upload_video_btn = tk.Button(root, text="Upload Video", command=upload_video, width=20)
upload_video_btn.pack(pady=5)

webcam_btn = tk.Button(root, text="Start Webcam", command=start_webcam_thread, width=20)
webcam_btn.pack(pady=5)

exit_btn = tk.Button(root, text="Exit", command=root.quit, width=20)
exit_btn.pack(pady=5)

root.mainloop()
