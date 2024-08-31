from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import seaborn as sns
from sklearn.metrics import log_loss
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm.notebook import tqdm
XCEPTION_MODEL = './deepfakemodelspackages/xception-b5690688.pth'

# test_dir = "./test/real"

# test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
# len(test_videos)

# Install packages
# !pip install ./deepfakemodelspackages/Pillow-6.2.1-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/munch-2.5.0-py2.py3-none-any.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/numpy-1.17.4-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ -f ./ --no-index
# !pip install ./deepfakemodelspackages/six-1.13.0-py2.py3-none-any.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/torchvision-0.4.2-cp36-cp36m-manylinux1_x86_64.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/tqdm-4.40.2-py2.py3-none-any.whl -f ./ --no-index
# !pip install ./deepfakemodelspackages/dlib-19.19.0/dlib-19.19.0/ -f ./ --no-index

# Model Integration 
## xception.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

# pretrained_settings = {
#     'xception': {
#         'imagenet': {
#             'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
#             'input_space': 'RGB',
#             'input_size': [3, 299, 299],
#             'input_range': [0, 1],
#             'mean': [0.5, 0.5, 0.5],
#             'std': [0.5, 0.5, 0.5],
#             'num_classes': 1000,
#             'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
#         }
#     }
# }


# class SeparableConv2d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
#         super(SeparableConv2d,self).__init__()

#         self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
#         self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x


# class Block(nn.Module):
#     def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
#         super(Block, self).__init__()

#         if out_filters != in_filters or strides!=1:
#             self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
#             self.skipbn = nn.BatchNorm2d(out_filters)
#         else:
#             self.skip=None

#         self.relu = nn.ReLU(inplace=True)
#         rep=[]

#         filters=in_filters
#         if grow_first:
#             rep.append(self.relu)
#             rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
#             rep.append(nn.BatchNorm2d(out_filters))
#             filters = out_filters

#         for i in range(reps-1):
#             rep.append(self.relu)
#             rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
#             rep.append(nn.BatchNorm2d(filters))

#         if not grow_first:
#             rep.append(self.relu)
#             rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
#             rep.append(nn.BatchNorm2d(out_filters))

#         if not start_with_relu:
#             rep = rep[1:]
#         else:
#             rep[0] = nn.ReLU(inplace=False)

#         if strides != 1:
#             rep.append(nn.MaxPool2d(3,strides,1))
#         self.rep = nn.Sequential(*rep)

#     def forward(self,inp):
#         x = self.rep(inp)

#         if self.skip is not None:
#             skip = self.skip(inp)
#             skip = self.skipbn(skip)
#         else:
#             skip = inp

#         x+=skip
#         return x


# class Xception(nn.Module):

#     def __init__(self, num_classes=1000):
#         super(Xception, self).__init__()
#         self.num_classes = num_classes

#         self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(32,64,3,bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         #do relu here

#         self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
#         self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
#         self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

#         self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

#         self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

#         self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

#         self.conv3 = SeparableConv2d(1024,1536,3,1,1)
#         self.bn3 = nn.BatchNorm2d(1536)

#         #do relu here
#         self.conv4 = SeparableConv2d(1536,2048,3,1,1)
#         self.bn4 = nn.BatchNorm2d(2048)

#         self.fc = nn.Linear(2048, num_classes)

#         # #------- init weights --------
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()
#         # #-----------------------------

#     def features(self, input):
#         x = self.conv1(input)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#         x = self.block12(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         return x

#     def logits(self, features):
#         x = self.relu(features)

#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.view(x.size(0), -1)
#         x = self.last_linear(x)
#         return x

#     def forward(self, input):
#         x = self.features(input)
#         x = self.logits(x)
#         return x


## models.py
import os
import argparse


import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception
import math
import torchvision



## transform.py
from torchvision import transforms

## detect_from_video.py
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm.notebook import tqdm


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init







import os
import argparse


import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision





#################################################################
# torch.nn.Module.dump_patches = True

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=True)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            #'/home/ondyari/.torch/models/xception-b5690688.pth')
            XCEPTION_MODEL)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

model_path = './deepfakemodelspackages/faceforensics_models/faceforensics++_models_subset/full/xception/full_raw.p'
model = torch.load(model_path, map_location=torch.device('cpu'))

def preprocess_image(image, cuda=False):
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=False):
    # Preprocess
    preprocessed_image = preprocess_image(image, cuda)

    # Model prediction
    output = model(preprocessed_image)
    output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output

def video_file_frame_pred(video_path, model,
                          start_frame=0, end_frame=300,
                          cuda=False, n_frames=64):
    
    pred_frames = [int(round(x)) for x in np.linspace(start_frame, end_frame, n_frames)]
    predictions = []
    outputs = []
    # print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1
        if frame_num in pred_frames:
            height, width = image.shape[:2]
            # 2. Detect with dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0]
                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]

                # Actual prediction using our model
                prediction, output = predict_with_model(cropped_face, model,
                                                        cuda=cuda)
                predictions.append(prediction)
                outputs.append(output)
                # ------------------------------------------------------------------
        if frame_num >= end_frame:
            break
    # Figure out how to do this with torch
    preds_np = [x.detach().cpu().numpy()[0][1] for x in outputs]
    if len(preds_np) == 0:
        return predictions, outputs, 0.5, 0.5, 0.5
    try:
        mean_pred = np.mean(preds_np)
    except:
        # couldnt find faces
        mean_pred = 0.5
    min_pred = np.min(preds_np)
    max_pred = np.max(preds_np)
    # return predictions, outputs, mean_pred, min_pred, max_pred
    return mean_pred
# def predict_on_video_set(videos):
#     predictions = []
    
#     for i in range(len(videos)):
#         filename = videos[i]
#         prediction, outputs, mean_pred, min_pred, max_pred = video_file_frame_pred(os.path.join(test_dir, filename), model)
#         predictions.append(mean_pred)
        
#     return predictions


app = FastAPI()
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            video_path = temp_file.name
        
        # Run the prediction logic
        detection_result = video_file_frame_pred(video_path, model)
        temp = ""
        if detection_result < 0.5:
            temp = "Real"
        elif detection_result > 0.5:
            temp = "Fake"
        else:
            temp = "Ambiguous"
    
        
        # Cleanup temporary file
        os.remove(video_path)

        return JSONResponse(content={"result": temp})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
