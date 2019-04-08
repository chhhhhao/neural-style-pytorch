import torch as t
import torchvision as tv
import utils
from modifiedVGG import Vgg16
from torch.nn import functional as F
class Config(object):
    image_size=512
    style_path='style_images/style.jpg'
    content_path='content_images/content.jpg'
    combined_path='combined_images'

    content_weight=1
    style_weight=1000
    lr=1e-3
    epoches=7000
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

def train():
    cfg = Config()
    vgg = Vgg16().to(cfg.device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    # 固定网络的参数
    content = utils.get_image_data(cfg.content_path,cfg.image_size).to(cfg.device)
    style = utils.get_image_data(cfg.style_path,cfg.image_size).to(cfg.device)
    target = content.clone().requires_grad_(True)


    content_features = vgg(content)
    style_features = vgg(style)
    gram_styles = [utils.gram_matrix(x).requires_grad_(False) for x in style_features]
        # 注意要使style——gram的requires_grad置于False，F.mse_loss要求
    optimizer = t.optim.Adam([target],lr=cfg.lr)
    for epoch in range(cfg.epoches):
        target_features = vgg(target)
        content_loss = F.mse_loss(target_features.relu3_3,content_features.relu3_3.requires_grad_(False))

        style_loss=0.
        for tar,gram_style in zip(target_features,gram_styles):
            tar_gram = utils.gram_matrix(tar)
            style_loss += F.mse_loss(tar_gram,gram_style)
        total_loss = cfg.content_weight*content_loss + cfg.style_weight*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print("iteration:{}  Content loss:{:.4f},Style loss:{:.4f},Total loss:{:.4f}".format(epoch+1,content_loss.item(),style_loss.item(),total_loss.item()))

    denorm = tv.transforms.Normalize([-2.12,-2.04,-1.80],[4.37,4.46,4.44])
    target = denorm(target.squeeze().to('cpu')).clamp_(min = 0,max = 1)
    tv.utils.save_image(target,cfg.combined_path + '/output '+str(cfg.content_weight/cfg.style_weight)+'.png')

train()
