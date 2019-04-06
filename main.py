import torch as t
import torchvision as tv
import utils
from modifiedVGG import Vgg16
from torch.nn import functional as F
class Config(object):
    image_size=256
    style_path='style_images/'
    content_path='content_images/'
    combined_path='combined_images/'

    content_weight=1e5
    style_weight=1e10
    lr=1e-3
    epoches=20
    device = torch.device("cuda" if use_cuda else "cpu")

def train():
    cfg = Config()
    vgg = Vgg16().to(device).eval()
    for param in vgg,parameters():
        param.requires_grad = False
    # 固定网络的参数
    content = utils.get_image_data(cfg.content_path).to(device)
    style = utils.get_image_data(cfg.style_path).to(device)
    target = content.clone().requires_grad_(True)

    content_features = vgg(content)
    style_features = vgg(style)
    gram_styles = [utils.gram_matrix(x) for x in style_features]

    optimizer = torch.optim.Adam([target],lr=cfg.lr)

    for epoch in range(cfg.epoches):
        target_features = vgg(target)
        content_loss = F.mse_loss(target_features.relu3_3,content_features.relu3_3)

        style_loss=0.
        for tar,gram_style in zip(target_features,gram_styles):
            tar_gram = utils.gram_matrix(tar)
            style_loss += F.mseloss(tar_gram,gram_style)
        total_loss = cfg.content_weight*content_loss + cfg.style_weight*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("iteration:{},Content loss:{:.4f},Style loss:{:.4f},Total loss:{:.4f}"
        .format(epoch+1,content_loss.item(),style_loss.item(),total_loss.item()))
    target = target.clamp(min = 0,max = 1).squeeze()
    tv.utils.save_image(target,cfg.combined_path+'output.png')
