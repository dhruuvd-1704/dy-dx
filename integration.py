import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
frames_per_video = 100
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 150
from torchvision.transforms import Normalize

test_dir="F:/dataset/fake"
test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
from pytorchcv.model_provider import get_model
model = get_model("xception", pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

class Pooling(nn.Module):
  def __init__(self):
    super(Pooling, self).__init__()
    
    self.p1 = nn.AdaptiveAvgPool2d((1,1))
    self.p2 = nn.AdaptiveMaxPool2d((1,1))

  def forward(self, x):
    x1 = self.p1(x)
    x2 = self.p2(x)
    return (x1+x2) * 0.5

model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()
    self.l = nn.Linear(in_f, 512)
    self.d = nn.Dropout(0.5)
    self.o = nn.Linear(512, out_f)
    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)
    self.r = nn.ReLU()

  def forward(self, x):
    x = self.f(x)
    x = self.b1(x)
    x = self.d(x)

    x = self.l(x)
    x = self.r(x)
    x = self.b2(x)
    x = self.d(x)

    out = self.o(x)
    return out

class FCN(torch.nn.Module):
  def __init__(self, base, in_f):
    super(FCN, self).__init__()
    self.base = base
    self.h1 = Head(in_f, 1)
  
  def forward(self, x):
    x = self.base(x)
    return self.h1(x)

net = []
model = FCN(model, 2048)
# model = model.cuda()
model.load_state_dict(torch.load('F:/source/xception/model.pth', map_location=torch.device('cpu')))
net.append(model)

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces_from_video(video_path, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    frame_number = 0
    face_number = 0
    flag = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterate over detected faces and save them
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_filename = os.path.join(output_folder, f"frame{frame_number}_face{face_number}.jpg")
            cv2.imwrite(face_filename, face)
            face_number += 1
            if(face_number >= 100): 
                flag = 1 
                break
        if (flag):
            break;
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extraction complete. {face_number} faces extracted.")


def predict_on_video(video_path, batch_size):
    try:
        # Directory to temporarily save extracted faces
        temp_faces_directory = "./temp_faces"
        os.makedirs(temp_faces_directory, exist_ok=True)

        # Extract faces from the video using the custom function
        extract_faces_from_video(video_path, temp_faces_directory)

        # List all face images in the directory
        face_files = os.listdir(temp_faces_directory)

        if not face_files:
            print("No faces found in video:", video_path)
            return 0.5

        # Initialize array for batch processing
        x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
        n = 0

        for face_file in face_files:
            face_path = os.path.join(temp_faces_directory, face_file)
            face = cv2.imread(face_path)

            # Preprocess face as needed for your model
            resized_face = isotropically_resize_image(face, input_size)
            resized_face = make_square_image(resized_face)

            if n < batch_size:
                x[n] = resized_face
                n += 1
            else:
                print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

        if n > 0:
            x = torch.tensor(x).float()
            x = x.permute((0, 3, 1, 2))

            for i in range(len(x)):
                x[i] = normalize_transform(x[i] / 255.)

            # Make predictions with the model
            with torch.no_grad():
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.squeeze())
                return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    finally:
        # Clean up temporary faces directory
        if os.path.exists(temp_faces_directory):
            for file in os.listdir(temp_faces_directory):
                file_path = os.path.join(temp_faces_directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(temp_faces_directory)

    return 0.5

def predict_on_video_set(videos):
    predictions = []
    for filename in videos:
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        predictions.append(y_pred)

    return predictions

model.eval()
predictions = predict_on_video_set(test_videos)
