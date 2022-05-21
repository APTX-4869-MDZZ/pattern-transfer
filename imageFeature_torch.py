import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureModel(nn.Module):
  def __init__(self, model_name='VGG16'):
    super(FeatureModel, self).__init__()
    self.model_name = model_name
    if model_name == 'ResNet50':
      backbone = models.resnet50(pretrained=True)
    elif model_name == 'InceptionV3':
      backbone = models.inception_v3(pretrained=True)
    else:
      backbone = models.vgg16(pretrained=True)
    features = list(backbone.children())[:-2]
    self.model = nn.Sequential(*features)
    self.avgPooling = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = self.model(x)
    x = self.avgPooling(x)
    return x

class ImageFeature():
  def __init__(self, device, model_name='VGG16'):
    self.model = FeatureModel(model_name)
    self.device = device
    self.model.to(device)

    pretrained_size = 224
    if model_name == 'InceptionV3':
      pretrained_size = 299
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]
    self.transform = transforms.Compose([
      transforms.Resize(pretrained_size),
      transforms.ToTensor(),
      transforms.Normalize(mean = pretrained_means, 
                            std = pretrained_stds)
    ])

  def get_image_feature(self, img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = self.transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(self.device)
    with torch.no_grad():
      out = self.model(batch_t)
      return torch.squeeze(out).cpu().numpy()

if __name__ == '__main__':
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  imageProcessor = ImageFeature(device, 'VGG16')
  imageProcessor.get_image_feature('top_images/Q220/0.jpeg')
