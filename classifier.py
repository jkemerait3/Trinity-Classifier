import numpy as np
import uproot

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

PATH_TO_NEG_DATA = "/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstratorDataAnalysis/event_cleaning/Output/"
PATH_TO_POS_DATA = None

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class CNN(nn.m)

