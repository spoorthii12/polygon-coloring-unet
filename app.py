import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gradio as gr

MODEL_PATH = "best_model.pth"
COLOR_LIST = ["red","green","blue","yellow","orange","purple","cyan","magenta"]
IMG_SIZE = 128
NUM_COLORS = len(COLOR_LIST)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_ch=3+NUM_COLORS
        self.inc=DoubleConv(in_ch,64)
        self.down1=torch.nn.Sequential(torch.nn.MaxPool2d(2),DoubleConv(64,128))
        self.down2=torch.nn.Sequential(torch.nn.MaxPool2d(2),DoubleConv(128,256))
        self.down3=torch.nn.Sequential(torch.nn.MaxPool2d(2),DoubleConv(256,512))
        self.up1=torch.nn.ConvTranspose2d(512,256,2,2);self.upc1=DoubleConv(512,256)
        self.up2=torch.nn.ConvTranspose2d(256,128,2,2);self.upc2=DoubleConv(256,128)
        self.up3=torch.nn.ConvTranspose2d(128,64,2,2); self.upc3=DoubleConv(128,64)
        self.outc=torch.nn.Conv2d(64,3,1)
    def forward(self,x,c):
        b,h,w=x.size(0),x.size(2),x.size(3)
        c=c.unsqueeze(2).unsqueeze(3).expand(b,NUM_COLORS,h,w)
        x=torch.cat([x,c],1)
        x1=self.inc(x);x2=self.down1(x1);x3=self.down2(x2);x4=self.down3(x3)
        u1=self.up1(x4);u1=torch.cat([u1,x3],1);u1=self.upc1(u1)
        u2=self.up2(u1);u2=torch.cat([u2,x2],1);u2=self.upc2(u2)
        u3=self.up3(u2);u3=torch.cat([u3,x1],1);u3=self.upc3(u3)
        return torch.tanh(self.outc(u3))

device=torch.device('cpu')
model=UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.eval()

preprocess=T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
def color_to_onehot_vec(color):
    vec=[0]*NUM_COLORS;vec[COLOR_LIST.index(color)]=1
    return torch.tensor(vec).float()

def inference(img,color):
    x=preprocess(img).unsqueeze(0)
    c=color_to_onehot_vec(color).unsqueeze(0)
    with torch.no_grad(): out=model(x,c)
    out=((out.squeeze(0).permute(1,2,0).numpy()+1)/2).clip(0,1)
    return out

gr.Interface(fn=inference,
             inputs=[gr.inputs.Image(type="pil",label="Polygon Image"),
                     gr.inputs.Dropdown(COLOR_LIST,label="Color")],
             outputs=gr.outputs.Image(type="numpy",label="Colored Polygon"),
             title="Polygon Coloring UNet",
             description="Upload a polygon and select a color."
).launch()
