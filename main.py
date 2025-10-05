# from ultralytics import YOLO
# import math
# import cv2
# import cvzone
# import torch
# from image_to_text import predict_number_plate
# from transformers import VisionEncoderDecoderModel
# from transformers import TrOCRProcessor
# from paddleocr import PaddleOCR

# cap = cv2.VideoCapture("videos/22.mp4")  # For videos

# model = YOLO("C:/project1/myproject/runs/detect/train2/weights/best.pt") # after training update the location of best.pt

# device = torch.device("cpu") # change to cuda for windows gpu or keep it as cpu

# classNames = ["with helmet", "without helmet", "rider", "number plate"]
# num = 0
# old_npconf = 0

# # grab the width, height, and fps of the frames in the video stream.
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # initialize the FourCC and a video writer object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))


# ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory

# while True:
#     success, img = cap.read()
#     # Check if the frame was read successfully
#     if not success:
#         break
#     new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = model(new_img, stream=True, device="cpu")
#     for r in results:
#         boxes = r.boxes
#         li = dict()
#         rider_box = list()
#         xy = boxes.xyxy
#         confidences = boxes.conf
#         classes = boxes.cls
#         new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
#         try:
#             new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
#             # Get the indices of the rows where the value in column 1 is equal to 5.
#             indices = torch.where(new_boxes[:, -1] == 2)
#             # Select the rows where the mask is True.
#             rows = new_boxes[indices]
#             # Add rider details in the list
#             for box in rows:
#                 x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 rider_box.append((x1, y1, x2, y2))
#         except:
#             pass
#         for i, box in enumerate(new_boxes):
#             # Bounding box
#             x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             # Confidence
#             conf = math.ceil((box[4] * 100)) / 100
#             # Class Name
#             cls = int(box[5])
#             if classNames[cls] == "without helmet" and conf >= 0.5 or classNames[cls] == "rider" and conf >= 0.45 or \
#                     classNames[cls] == "number plate" and conf >= 0.5:
#                 if classNames[cls] == "rider":
#                     rider_box.append((x1, y1, x2, y2))
#                 if rider_box:
#                     for j, rider in enumerate(rider_box):
#                         if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and \
#                                 y2 <= rider_box[j][3]:
#                             # highlight or outline objects detected by object detection models
#                             cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
#                             cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
#                                                offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
#                             li.setdefault(f"rider{j}", [])
#                             li[f"rider{j}"].append(classNames[cls])
#                             if classNames[cls] == "number plate":
#                                 npx, npy, npw, nph, npconf = x1, y1, w, h, conf
#                                 crop = img[npy:npy + h, npx:npx + w]
#                         if li:
#                             for key, value in li.items():
#                                 if key == f"rider{j}":
#                                     if len(list(set(li[f"rider{j}"]))) == 3:
#                                         try:
#                                             # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) # for easy ocr
#                                             vechicle_number, conf = predict_number_plate(crop, ocr)
#                                             if vechicle_number and conf:
#                                                 cvzone.putTextRect(img, f"{vechicle_number} {round(conf*100, 2)}%",
#                                                                    (x1, y1 - 50), scale=1.5, offset=10,
#                                                                    thickness=2, colorT=(39, 40, 41),
#                                                                    colorR=(105, 255, 255))
#                                         except Exception as e:
#                                             print(e)
#         # Display the frame
#         output.write(img)
#         cv2.imshow('Video', img)
#         li = list()
#         rider_box = list()

#         # Exit the program if the 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             output.release()
#             break






# import cv2
# import torch
# from ultralytics import YOLO
# import cvzone
# from image_to_text import predict_number_plate
# from paddleocr import PaddleOCR
# import time

# # Load YOLO Model
# model = YOLO("C:/project1/myproject/runs/detect/train2/weights/best.pt")

# # Force CPU usage (change to 'cuda' if GPU is available)
# device = torch.device("cpu")

# # Class labels
# classNames = ["with helmet", "without helmet", "rider", "number plate"]

# # Open video file
# cap = cv2.VideoCapture("videos/22.mp4")

# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# # Setup video writer
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_image_shape="3, 32, 64")

# # Frame skipping to reduce processing load
# frame_skip = 2  
# frame_count = 0

# while True:
#     success, img = cap.read()
#     if not success:
#         break  # Stop if video ends

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue  # Skip frames to improve speed

#     # Convert to RGB for YOLO
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Run YOLO only once per frame
#     start_time = time.time()
#     results = model(img_rgb, stream=True, device='cpu')
#     end_time = time.time()

#     # Check inference time
#     inference_time = end_time - start_time
#     print(f"Inference Time: {inference_time:.2f}s")

#     for r in results:
#         boxes = r.boxes  # Get all detected objects

#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 🔥 FIXED LINE
#             conf = round(float(box.conf[0]), 2)
#             cls = int(box.cls[0])

#             # Skip low-confidence detections
#             if conf < 0.5:
#                 continue

#             # Draw bounding boxes
#             cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=5, colorR=(255, 0, 0))
#             cvzone.putTextRect(img, f"{classNames[cls].upper()} ({conf})", (x1, y1 - 10),
#                                scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

#             # Number Plate OCR Processing
#             if classNames[cls] == "number plate":
#                 crop = img[y1:y2, x1:x2]
#                 if crop.size > 0:
#                     vechicle_number, conf = predict_number_plate(crop, ocr)
#                     if vechicle_number:
#                         cvzone.putTextRect(img, f"{vechicle_number} ({round(conf * 100, 2)}%)",
#                                            (x1, y1 - 50), scale=1.5, offset=10,
#                                            thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))

#     # Clear GPU memory (only needed for torch)
#     torch.cuda.empty_cache()

#     output.write(img)
#     cv2.imshow('Helmet & Number Plate Detection', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# output.release()
# cv2.destroyAllWindows()







# write code

import cv2
import torch
from ultralytics import YOLO
import cvzone
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR
import time
from sendemail import send_email_alert  # ✅ Import email function


# Load YOLO Model
model = YOLO("C:/project1/myproject/runs/detect/train2/weights/best.pt")

# Force CPU usage (change to 'cuda' if GPU is available)
device = torch.device("cpu")

# Class labels
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_image_shape="3, 32, 64")

# Function to process an image file
def process_image(image_path):
    print(f"Processing Image: {image_path}")  # Debugging log
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    processed_img = process_frame(img)
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a video file
def process_video(video_path):
    print(f"Processing Video: {video_path}")  # Debugging log
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break  # Stop if video ends

        img = process_frame(img)  # Process frame
        cv2.imshow("Processed Video", img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start webcam processing
def start_webcam():
    print("Starting Webcam...")  # Debugging log
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        img = process_frame(img)  # Process webcam frame
        cv2.imshow("Webcam - Helmet & Number Plate Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# this is original
# Function to process frames (used for video, webcam, and image)
# def process_frame(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = model(img_rgb, stream=True, device='cpu')

#     for r in results:
#         boxes = r.boxes  # Get all detected objects
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 🔥 FIXED LINE
#             conf = round(float(box.conf[0]), 2)
#             cls = int(box.cls[0])

#             # Skip low-confidence detections
#             if conf < 0.5:
#                 continue

#             # Draw bounding boxes
#             cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=5, colorR=(255, 0, 0))
#             cvzone.putTextRect(img, f"{classNames[cls].upper()} ({conf})", (x1, y1 - 10),
#                                scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

#             # Number Plate OCR Processing
#             if classNames[cls] == "number plate":
#                 crop = img[y1:y2, x1:x2]
#                 if crop.size > 0:
#                     vechicle_number, conf = predict_number_plate(crop, ocr)
#                     if vechicle_number:
#                         cvzone.putTextRect(img, f"{vechicle_number} ({round(conf * 100, 2)}%)",
#                                            (x1, y1 - 50), scale=1.5, offset=10,
#                                            thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
    
#     return img

import os

def process_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, stream=True, device='cpu')

    detected_plates = []  # Store detected plates
    rider_without_helmet = False  # Flag to check helmet violation

    for r in results:
        boxes = r.boxes  # Get all detected objects
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            if conf < 0.5:
                continue  # Skip low-confidence detections

            label = classNames[cls]
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=5, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{label.upper()} ({conf})", (x1, y1 - 10),
                               scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

            # ✅ Detect helmet violation
            if label == "without helmet":
                rider_without_helmet = True  

            # ✅ Detect number plate and extract text
            if label == "number plate":
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    vechicle_number, conf = predict_number_plate(crop, ocr)
                    if vechicle_number:
                        detected_plates.append(vechicle_number)
                        cvzone.putTextRect(img, f"{vechicle_number} ({round(conf * 100, 2)}%)",
                                           (x1, y1 - 50), scale=1.5, offset=10,
                                           thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))


    # ✅ If a rider is without a helmet AND a number plate is detected, send alert
    if rider_without_helmet and detected_plates:
        full_image_path = f"detected_violations/{vechicle_number}.jpg"
        os.makedirs("detected_violations", exist_ok=True)  # Ensure folder exists
        cv2.imwrite(full_image_path, img)
        print(f"✅ Saved Full Violation Image: {full_image_path}")

        for plate_number in detected_plates:
            send_email_alert(plate_number, full_image_path)  # Send email with full image
            # send_sms_alert(plate_number)  # Send SMS alert

    return img












# # adding email



# import cv2
# import torch
# from ultralytics import YOLO
# import cvzone
# from image_to_text import predict_number_plate
# from paddleocr import PaddleOCR
# import time
# import smtplib
# import ssl
# import os
# import cv2
# from email.message import EmailMessage


# # Load YOLO Model
# model = YOLO("C:/project1/myproject/runs/detect/train2/weights/best.pt")

# # Force CPU usage (change to 'cuda' if GPU is available)
# device = torch.device("cpu")

# # Class labels
# classNames = ["with helmet", "without helmet", "rider", "number plate"]

# # Initialize PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_image_shape="3, 32, 64")

# # Function to process an image file
# def process_image(image_path):
#     print(f"Processing Image: {image_path}")  # Debugging log
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Could not load image.")
#         return

#     processed_img = process_frame(img)
#     cv2.imshow("Processed Image", processed_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Function to process a video file
# def process_video(video_path):
#     print(f"Processing Video: {video_path}")  # Debugging log
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         return

#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             break  # Stop if video ends

#         img = process_frame(img)  # Process frame
#         cv2.imshow("Processed Video", img)

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Function to start webcam processing
# def start_webcam():
#     print("Starting Webcam...")  # Debugging log
#     cap = cv2.VideoCapture(0)  # Open webcam

#     if not cap.isOpened():
#         print("Error: Could not access webcam.")
#         return

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         img = process_frame(img)  # Process webcam frame
#         cv2.imshow("Webcam - Helmet & Number Plate Detection", img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Function to process frames (used for video, webcam, and image)
# def process_frame(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = model(img_rgb, stream=True, device='cpu')

#     for r in results:
#         boxes = r.boxes  # Get all detected objects
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 🔥 FIXED LINE
#             conf = round(float(box.conf[0]), 2)
#             cls = int(box.cls[0])

#             # Skip low-confidence detections
#             if conf < 0.5:
#                 continue

#             # Draw bounding boxes
#             cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=5, colorR=(255, 0, 0))
#             cvzone.putTextRect(img, f"{classNames[cls].upper()} ({conf})", (x1, y1 - 10),
#                                scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

#             # Number Plate OCR Processing
#             if classNames[cls] == "number plate":
#                 crop = img[y1:y2, x1:x2]
#                 if crop.size > 0:
#                     vechicle_number, conf = predict_number_plate(crop, ocr)
#                     if vechicle_number:
#                         cvzone.putTextRect(img, f"{vechicle_number} ({round(conf * 100, 2)}%)",
#                                            (x1, y1 - 50), scale=1.5, offset=10,
#                                            thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255)) 
            
                        
#             # Check if the rider is without a helmet
#             for rider_box in boxes:
#                 rider_cls = int(rider_box.cls[0])
#                 if classNames[rider_cls] == "without helmet":
#                     # Save the image
#                     img_path = f"violations/{vechicle_number}.jpg"
#                     cv2.imwrite(img_path, crop)

#                     # Send Email Alert 🚨
#                     send_email_alert(vechicle_number, img_path)
                    
                        
#             # Email Configuration
#             EMAIL_SENDER = "abhaybagul360@gmail.com"
#             EMAIL_PASSWORD = "Abhay@123"
#             EMAIL_RECEIVER = "abhaybagul2003@gmail.com"

#             def send_email_alert(plate_number, image_path):
#                 """Send an email when a violation is detected."""
#                 subject = f"Helmet Violation Detected - {plate_number}"
#                 body = f"""
#                 A helmet violation has been detected.
#                 Number Plate: {plate_number}
#                 See attached image for reference.
#                 """ 

#                 msg = EmailMessage()
#                 msg["Subject"] = subject
#                 msg["From"] = EMAIL_SENDER
#                 msg["To"] = EMAIL_RECEIVER
#                 msg.set_content(body)

#                 # Attach image
#                 with open(image_path, "rb") as img:
#                     msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

#                 # Send Email
#                 context = ssl.create_default_context()
#                 with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#                     server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#                     server.send_message(msg)
                
#                 print(f"🚨 Email Alert Sent: {plate_number} 🚔")



                                    


    
#     return img


