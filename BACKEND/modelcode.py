

'''
# UTILITIES
from torchvision.models import ResNet18_Weights
import os, torch, cv2
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
EPOCHS = 10
SEQUENCE_LENGTH = 20
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




# LOAD DATA


class VideoDataset(Dataset):
  def __init__(self, root_dir):
    self.transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.486,0.455,0.406],[0.228,0.225,0.224])
    ])

    self.samples = []
    for labels,cls in enumerate(CLASSES):
      class_path = os.path.join(root_dir,cls)
      if not os.path.exists(class_path):
          continue
      for videofolder in os.listdir(class_path):
        video_folder = os.path.join(class_path,videofolder)
        if os.path.isdir(video_folder):
          self.samples.append((video_folder,labels))

  def extract_frames(self,video_folder):
    frame_files = sorted(os.listdir(video_folder))
    frames=[]

    for img_name in frame_files[:SEQUENCE_LENGTH]:
      img_path = os.path.join(video_folder,img_name)

      img = Image.open(img_path).convert("RGB")
      img = self.transform(img)
      frames.append(img)

    if len(frames) == 0:
      return None

    while len(frames) < SEQUENCE_LENGTH:
      frames.append(frames[-1]) # [3,224,244]
    return torch.stack(frames,dim = 0) # [16,3,224,224] ,dim = 0 for row-wise stacking

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    original_idx = idx

    while True:
        video_folder, label = self.samples[idx]
        frames = self.extract_frames(video_folder)

        if frames is not None:
            return frames, label

        idx = (idx + 1) % len(self.samples)

        # agar poora dataset ek round me fail ho gaya
        if idx == original_idx:
            # dummy black frames return karo
            dummy = torch.zeros(
                (SEQUENCE_LENGTH, 3, IMAGE_SIZE, IMAGE_SIZE)
            )
            return dummy, label



class ActionModel(nn.Module):

  def __init__(self):
    super().__init__()
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    self.cnn = nn.Sequential(*list(model.children())[:-1]) # children() pytorch ka ek function hai jo model ke sub-module/layes ko return karta hai

    for param in self.cnn.parameters():
      param.requires_grad = False

    for name,param in model.named_parameters():
      if "layer4" not in name:
        param.requires_grad = False
    self.lstm = nn.LSTM(512,256,batch_first = True) # 512 = input_size,256 = hidden_size
    self.fc = nn.Linear(256,len(CLASSES))

  def forward(self,x):
    b,t,c,h,w = x.size()
    # [8,16,3,224,224]

    x = x.view(b*t,c,h,w)
    # [128,3,224,224]

    feats = self.cnn(x)
    # [128,512,1,1]

    feats = feats.view(feats.size(0),-1)
    # [128,512]

    feats = feats.view(b,t,-1)
    # [8,16,512]

    out,_ = self.lstm(feats)
    # [8,16,256]

    return self.fc(out[:,-1])
    # [8,8]


train_dataset = "/content/drive/MyDrive/PYTORCH PROJECT DATASETS/archive (5)/HMDB51"
dataset = VideoDataset(train_dataset)
loader = DataLoader(dataset,batch_size = 8,shuffle = True,num_workers = 4)

model = ActionModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                             lr = 1e-4
                             )
model_path1 = f"/content/drive/MyDrive/Pytorch_project/11TH/trained_model"
os.makedirs(model_path1, exist_ok=True)
best_f1 = 0.0
for epoch in range(EPOCHS):
  print("-"*20)
  print(f"Epoch {epoch+1}/{EPOCHS}")
  y_true = []
  y_pred = []
  model.train()
  total_loss = 0.0
  for inputs,labels in loader:
    inputs, labels = inputs.to(DEVICE),labels.to(DEVICE)
    optimizer.zero_grad()
    output = model(inputs)
    pred = torch.argmax(output,dim = 1)
    loss = criterion(output,labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    y_true.extend(labels.detach().cpu().numpy().tolist())
    y_pred.extend(pred.detach().cpu().numpy().tolist())
  print(f"Loss :- {total_loss/len(loader):.4f}")
  print(f"The accuracy is :- {accuracy_score(y_true,y_pred)}")
  f1_scr = f1_score(y_true,y_pred,average = 'macro',zero_division = 0)
  print(f"The F1_score is :- {f1_scr}")

  model_path2 = os.path.join(model_path1,f"{epoch+1}.pth")
  if best_f1 < f1_scr:
    best_f1 = f1_scr
    torch.save(model.state_dict(),model_path2)
  print("")

model_path = "/content/drive/MyDrive/Pytorch_project/11TH/complete_trained_model.pth"
#os.makedirs(model_path,exist_ok = True)
torch.save(model.state_dict(),model_path)
print(f"Model saved at {model_path}")
'''