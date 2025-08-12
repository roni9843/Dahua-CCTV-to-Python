import cv2
import torch
import time
import os

# YOLOv5 মডেল লোড করা
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# তোমার working RTSP URL
rtsp_url = "rtsp://admin:abc12345@192.168.0.111:554/cam/realmonitor?channel=8&subtype=0"

# RTSP URL ব্যবহার করে VideoCapture ইনিশিয়ালাইজ করা
cap = cv2.VideoCapture(rtsp_url)

# কনফিডেন্স থ্রেশহোল্ড সেট করা
conf_threshold = 0.25

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

print("DVR stream opened successfully. Waiting for 5 seconds before taking a photo...")
time.sleep(5) # এখানে 5 সেকেন্ড অপেক্ষা করা হচ্ছে

print("Taking a single photo now...")

# ক্যামেরা থেকে একটি ফ্রেম নেওয়া
ret, frame = cap.read()
cap.release()  # একটি ছবি তোলার পর ক্যামেরা বন্ধ করে দেওয়া

if not ret:
    print("Error: Could not read frame from stream. Exiting...")
    exit()

# ছবিটির জন্য একটি ইউনিক নাম তৈরি করা (সময় অনুসারে)
timestamp = time.strftime("%Y%m%d-%H%M%S")
temp_image_path = f"captured_images/capture_{timestamp}.jpg"

# ছবি সেভ করার জন্য ফোল্ডার তৈরি করা
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# ফ্রেমটি ফাইলে সেভ করা
cv2.imwrite(temp_image_path, frame)
print(f"Captured a new image: {temp_image_path}")

try:
    # সেভ করা ফাইলটি ব্যবহার করে ডিটেকশন করা
    results = model(temp_image_path)
    detections = results.pred[0]
    
    human_detected = False
    
    # ডিটেকশনগুলো থেকে শুধুমাত্র 'person' এবং একটি নির্দিষ্ট conf-এর উপরেরগুলো নেওয়া
    for *box, conf, cls in detections:
        if int(cls) == 0 and conf > conf_threshold: 
            human_detected = True
            
            # সেভ করা ছবিটি লোড করে চিহ্নিত করার কাজ শুরু
            detected_frame = cv2.imread(temp_image_path)
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'Person: {conf:.2f}'
            cv2.putText(detected_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # মানুষ থাকলে চিহ্নিত করা ছবিটি সেভ করা
            marked_image_path = f"detected_images/detected_{timestamp}.jpg"
            if not os.path.exists("detected_images"):
                os.makedirs("detected_images")
            cv2.imwrite(marked_image_path, detected_frame)
            print(f"Human detected! Image saved as: {marked_image_path}")
            
            # টেম্পোরারি ফাইলটি ডিলিট করে দেওয়া
            os.remove(temp_image_path)

    if not human_detected:
        # মানুষ না থাকলে শুধু কনসোলে মেসেজ দেখানো এবং টেম্পোরারি ফাইলটি ডিলিট করা
        os.remove(temp_image_path)
        print(f"No human detected. Deleting image: {temp_image_path}")

except Exception as e:
    print(f"An error occurred during detection: {e}")
    # কোনো কারণে এরর হলে টেম্পোরারি ফাইলটি ডিলিট করে দেওয়া
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)