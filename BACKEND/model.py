import torch
import torch.nn as nn
from torchvision import models,transforms
from torchvision.models import ResNet18_Weights
from pathlib import Path
from PIL import Image
import cv2

CLASSES = [
    "walk",
    "talk",
    "stand",
    "sit",
    "smile",
    "eat",
    "drink",
    "laugh"
]
SEQUENCE_LENGTH = 60
IMAGE_SIZE = 224

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.486,0.455,0.406],[0.228,0.225,0.224])
    ])


class ActionModel(nn.Module):

  def __init__(self):
    super().__init__()
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    self.cnn = nn.Sequential(*list(model.children())[:-1])

    for param in self.cnn.parameters():
      param.requires_grad = False

    for name,param in model.named_parameters():
      if "layer4" not in name:
        param.requires_grad = False
    self.lstm = nn.LSTM(512,256,batch_first = True)
    self.fc = nn.Linear(256,len(CLASSES))

  def forward(self,x):

    b,t,c,h,w = x.size()
    x = x.view(b*t,c,h,w)
    feats = self.cnn(x)
    feats = feats.view(feats.size(0),-1)
    feats = feats.view(b,t,-1)
    out,_ = self.lstm(feats)
    return self.fc(out[:,-1])

model = ActionModel()
Base_dir = Path(__file__).resolve().parent.parent
state_dict = torch.load(Base_dir/"MODEL"/"10.pth",map_location="cpu")
model.load_state_dict(state_dict)
# print(model.parameters())


def extract_frame(video_path):
  
  cap = cv2.VideoCapture(video_path)
  SEQUENCE_LENGTH = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  print(f"Total Frames: {SEQUENCE_LENGTH}")
  frames = []

  while len(frames)<SEQUENCE_LENGTH:
    ret,frame = cap.read()

    if not ret:
      break
    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = transform(img)
    frames.append(img)

    

  cap.release()
  if len(frames) == 0:
      raise ValueError("No frames extracted from video")
    
  while len(frames)<SEQUENCE_LENGTH:
    frames.append(frames[-1])

  frames = torch.stack(frames)
  frames = frames.unsqueeze(0)
  return frames
    


def inference(inpvideo_location):
    
    frames = extract_frame(inpvideo_location)
    model.eval()

    with torch.no_grad():
      # img_video = transform(Image.open(inpvideo_location)).unsqueeze(0)
      output = model(frames)
      probabilities = torch.softmax(output,dim=1)
      pred_class = torch.argmax(probabilities,dim=1)
      conf = probabilities[0][pred_class]

    return pred_class.item(),conf.item()
