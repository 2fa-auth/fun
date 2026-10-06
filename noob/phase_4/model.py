#!/home/client/Documents/fun/venv/bin/python3
import torch.nn as nn
import det 

class BACKBONEModel(nn.Module): 
  def __init__(self, in_chan, out_features):
    super().__init__()
    
    # сверточная сеть BACKBONE:
    self.darknet = nn.Sequential(
    nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(192, 128, kernel_size=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 256, kernel_size=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(512, 256, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(512, 256, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(512, 256, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(512, 256, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(512, 512, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(1024, 512, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(1024, 512, kernel_size=1), nn.LeakyReLU(0.1),
    nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            
    nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
    nn.LeakyReLU(0.1),
            
    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
    nn.LeakyReLU(0.1),
    )
        
    self.fcs = nn.Sequential(
    nn.Flatten(),
    nn.Linear(1024 * det.X_GRID * det.Y_GRID, 4096),
    nn.LeakyReLU(0.1),
    nn.Dropout(0.5),
            
    nn.Linear(4096, out_features) # out 1470 
    )

  def forward(self, x): 
    out = self.darknet(x)
    out = self.fcs(out)
    return out.view(-1, 7, 7, det.NUM_CLASSES)