import requests
from tqdm import tqdm
import os
import gdown
import torch



def download(id, fname, root=os.path.expanduser('~/.cache/videon')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination

_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'

def load_i3d_pretrained(device=torch.device('cpu')):
    from .models.i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    #filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    filepath = os.path.join('store_pth', 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d