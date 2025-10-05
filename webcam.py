# this is original
# import cv2
# import tkinter as tk
# from tkinter import filedialog
# import threading
# import main       # Import main.py to use its functions


# # Function to update the status message
# def update_status(message):
#     status_label.config(text=message)
#     root.update_idletasks()

# def upload_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
#     if file_path:
#         update_status("Processing Image...")
#         threading.Thread(target=process_image_with_status, args=(file_path,)).start()

# def upload_video():
#     file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
#     if file_path:
#         update_status("Processing Video...")
#         threading.Thread(target=process_video_with_status, args=(file_path,)).start()

# def start_webcam_thread():
#     update_status("Starting Webcam...")
#     threading.Thread(target=process_webcam_with_status).start()

# # Wrappers for processing with status updates
# def process_image_with_status(image_path):
#     main.process_image(image_path)
#     update_status("Image Processing Complete!")

# def process_video_with_status(video_path):
#     main.process_video(video_path)
#     update_status("Video Processing Complete!")

# def process_webcam_with_status():
#     main.start_webcam()
#     update_status("Webcam Closed.")

# # UI Setup
# root = tk.Tk()
# root.title("Helmet & Number Plate Detection")
# root.geometry("400x300")

# title_label = tk.Label(root, text="Helmet & Number Plate Detection", font=("Arial", 14, "bold"))
# title_label.pack(pady=10)

# upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20)
# upload_image_btn.pack(pady=5)

# upload_video_btn = tk.Button(root, text="Upload Video", command=upload_video, width=20)
# upload_video_btn.pack(pady=5)

# webcam_btn = tk.Button(root, text="Start Webcam", command=start_webcam_thread, width=20)
# webcam_btn.pack(pady=5)

# status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10, "italic"), fg="blue")
# status_label.pack(pady=10)

# exit_btn = tk.Button(root, text="Exit", command=root.quit, width=20)
# exit_btn.pack(pady=5)

# root.mainloop()

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import main  # Import main.py to use its functions

# Function to update the status message
def update_status(message):
    status_label.config(text=message)
    root.update_idletasks()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        update_status("📷 Processing Image... please wait")
        threading.Thread(target=process_image_with_status, args=(file_path,)).start()
    else:
        messagebox.showwarning("Warning", "No image selected!")

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        update_status("🎥 Processing Video...")
        threading.Thread(target=process_video_with_status, args=(file_path,)).start()
    else:
        messagebox.showwarning("Warning", "No video selected!")

def start_webcam_thread():
    update_status("📹 Starting Webcam...")
    threading.Thread(target=process_webcam_with_status, daemon=True).start()

# Wrappers for processing with status updates
def process_image_with_status(image_path):
    main.process_image(image_path)
    update_status("✅ Image Processing Complete!")

def process_video_with_status(video_path):
    main.process_video(video_path)
    update_status("✅ Video Processing Complete!")

def process_webcam_with_status():
    main.start_webcam()
    update_status("🔴 Webcam Closed.")


# UI Setup
root = tk.Tk()
root.title("Helmet & Number Plate Detection")
root.geometry("400x350")
root.resizable(False, False)

title_label = tk.Label(root, text="🚦 Helmet & Number Plate Detection", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

upload_image_btn = tk.Button(root, text="📸 Upload Image", command=upload_image, width=25, height=2, bg="#4CAF50", fg="white")
upload_image_btn.pack(pady=5)

upload_video_btn = tk.Button(root, text="🎬 Upload Video", command=upload_video, width=25, height=2, bg="#008CBA", fg="white")
upload_video_btn.pack(pady=5)

webcam_btn = tk.Button(root, text="📹 Start Webcam", command=start_webcam_thread, width=25, height=2, bg="#FF9800", fg="white")
webcam_btn.pack(pady=5)

status_label = tk.Label(root, text="Status: OK", font=("Arial", 10, "italic"), fg="blue")
status_label.pack(pady=10)


exit_btn = tk.Button(root, text="❌ Exit", command=root.quit, width=25, height=2, bg="red", fg="white")
exit_btn.pack(pady=5)

root.mainloop()


import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import main  # Import main.py to use its functions

# Function to update the status message
def update_status(message):
    status_label.config(text=message)
    root.update_idletasks()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        update_status("Processing Image...")
        threading.Thread(target=process_image_with_status, args=(file_path,)).start()
    else:
        messagebox.showwarning("Warning", "No image selected!")

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        update_status("Processing Video...")
        threading.Thread(target=process_video_with_status, args=(file_path,)).start()
    else:
        messagebox.showwarning("Warning", "No video selected!")

def start_webcam_thread():
    update_status("Starting Webcam...")
    threading.Thread(target=process_webcam_with_status, daemon=True).start()

# Wrappers for processing with status updates
def process_image_with_status(image_path):
    main.process_image(image_path)
    update_status("Image Processing Complete!")

def process_video_with_status(video_path):
    main.process_video(video_path)
    update_status("Video Processing Complete!")

def process_webcam_with_status():
    main.start_webcam()
    update_status("Webcam Closed.")

# UI Setup
root = tk.Tk()
root.title("Helmet & Number Plate Detection")
root.geometry("400x350")
root.resizable(False, False)

title_label = tk.Label(root, text="Helmet & Number Plate Detection", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, width=25, height=2, bg="#4CAF50", fg="white")
upload_image_btn.pack(pady=5)

upload_video_btn = tk.Button(root, text="Upload Video", command=upload_video, width=25, height=2, bg="#008CBA", fg="white")
upload_video_btn.pack(pady=5)

webcam_btn = tk.Button(root, text="Start Webcam", command=start_webcam_thread, width=25, height=2, bg="#FF9800", fg="white")
webcam_btn.pack(pady=5)

status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10, "italic"), fg="blue")
status_label.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", command=root.quit, width=25, height=2, bg="red", fg="white")
exit_btn.pack(pady=5)

root.mainloop()




# import cv2
# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import threading
# import main

# # Global variable to control webcam
# stop_webcam = False

# def process_image(image_path, status_label):
#     status_label.config(text="Processing Image...")
#     img = cv2.imread(image_path)
#     cv2.imshow("Selected Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     status_label.config(text="Image Processed")

# def process_video(video_path, status_label):
#     status_label.config(text="Processing Video...")
#     cap = cv2.VideoCapture(video_path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow("Video", frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     status_label.config(text="Video Processed")

# def start_webcam(status_label):
#     global stop_webcam
#     stop_webcam = False
#     cap = cv2.VideoCapture(0)
#     status_label.config(text="Webcam Running... Press 'Stop Webcam' to exit.")
#     while not stop_webcam:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imshow("Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     status_label.config(text="Webcam Stopped")

# def stop_webcam_stream():
#     global stop_webcam
#     stop_webcam = True

# def upload_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
#     if file_path:
#         threading.Thread(target=process_image, args=(file_path, status_label)).start()

# def upload_video():
#     file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
#     if file_path:
#         threading.Thread(target=process_video, args=(file_path, status_label)).start()

# def start_webcam_thread():
#     threading.Thread(target=start_webcam, args=(status_label,)).start()

# def exit_app():
#     if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
#         root.destroy()

# # UI Setup
# root = tk.Tk()
# root.title("Helmet & Number Plate Detection")
# root.geometry("500x400")
# root.configure(bg="#f0f0f0")

# # Title Label
# title_label = tk.Label(root, text="Helmet & Number Plate Detection", font=("Arial", 16, "bold"), fg="#333", bg="#f0f0f0")
# title_label.pack(pady=15)

# # Status Label
# status_label = tk.Label(root, text="Welcome!", font=("Arial", 12), fg="#555", bg="#f0f0f0")
# status_label.pack(pady=5)

# # Buttons
# upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20, font=("Arial", 12), bg="#4CAF50", fg="white")
# upload_image_btn.pack(pady=5)

# upload_video_btn = tk.Button(root, text="Upload Video", command=upload_video, width=20, font=("Arial", 12), bg="#2196F3", fg="white")
# upload_video_btn.pack(pady=5)

# webcam_btn = tk.Button(root, text="Start Webcam", command=start_webcam_thread, width=20, font=("Arial", 12), bg="#FF9800", fg="white")
# webcam_btn.pack(pady=5)

# stop_webcam_btn = tk.Button(root, text="Stop Webcam", command=stop_webcam_stream, width=20, font=("Arial", 12), bg="#FF5722", fg="white")
# stop_webcam_btn.pack(pady=5)

# exit_btn = tk.Button(root, text="Exit", command=exit_app, width=20, font=("Arial", 12), bg="#f44336", fg="white")
# exit_btn.pack(pady=10)

# root.mainloop()




# import cv2
# import tkinter as tk
# from tkinter import filedialog
# import threading
# import main  # Import main.py to use its functions

# # Global flag for stopping processes
# stop_processing = False  

# def update_status(message):
#     status_label.config(text=message)
#     root.update_idletasks()

# def upload_image():
#     file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
#     if file_path:
#         update_status("Processing Image...")
#         threading.Thread(target=process_image_with_status, args=(file_path,)).start()

# def upload_video():
#     file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
#     if file_path:
#         update_status("Processing Video...")
#         threading.Thread(target=process_video_with_status, args=(file_path,)).start()

# def start_webcam_thread():
#     update_status("Starting Webcam...")
#     threading.Thread(target=process_webcam_with_status).start()

# def stop_processing_func():
#     global stop_processing
#     stop_processing = True
#     update_status("Processing Stopped!")

# # Wrappers for processing with status updates
# def process_image_with_status(image_path):
#     global stop_processing
#     stop_processing = False  # Reset flag
#     main.process_image(image_path)
#     if not stop_processing:
#         update_status("Image Processing Complete!")

# def process_video_with_status(video_path):
#     global stop_processing
#     stop_processing = False  # Reset flag
#     main.process_video(video_path, stop_processing_func)
#     if not stop_processing:
#         update_status("Video Processing Complete!")

# def process_webcam_with_status():
#     global stop_processing
#     stop_processing = False  # Reset flag
#     main.start_webcam(stop_processing_func)
#     update_status("Webcam Closed.")

# # UI Setup
# root = tk.Tk()
# root.title("Helmet & Number Plate Detection")
# root.geometry("400x350")

# title_label = tk.Label(root, text="Helmet & Number Plate Detection", font=("Arial", 14, "bold"))
# title_label.pack(pady=10)

# upload_image_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20)
# upload_image_btn.pack(pady=5)

# upload_video_btn = tk.Button(root, text="Upload Video", command=upload_video, width=20)
# upload_video_btn.pack(pady=5)

# webcam_btn = tk.Button(root, text="Start Webcam", command=start_webcam_thread, width=20)
# webcam_btn.pack(pady=5)

# status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10, "italic"), fg="blue")
# status_label.pack(pady=10)

# stop_btn = tk.Button(root, text="Stop", command=stop_processing_func, width=20, fg="red")
# stop_btn.pack(pady=5)

# exit_btn = tk.Button(root, text="Exit", command=root.quit, width=20)
# exit_btn.pack(pady=5)

# root.mainloop()
