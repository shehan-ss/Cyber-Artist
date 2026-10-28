import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, max_size=1024):
    image = Image.open(img_path).convert('RGB')
    size = min(max_size, min(image.size))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
for param in vgg.parameters():
    param.requires_grad = False

def get_features(image, model):
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


content_img = load_image(r'C:\Users\ss\Desktop\ML\Cyber Artist\377c956e8d13099cd56bea2f114c8164.jpeg')
style_img1 = load_image(r'C:\Users\ss\Desktop\ML\Cyber Artist\van-gogh-starry-night.jpg')
style_img2 = load_image(r'C:\Users\ss\Desktop\ML\Cyber Artist\zhao-mengfu-twin-pines-level-distance.jpg')

content_features = get_features(content_img, vgg)
style_features1 = get_features(style_img1, vgg)
style_features2 = get_features(style_img2, vgg)

style_grams1 = {layer: gram_matrix(style_features1[layer]) for layer in style_features1}
style_grams2 = {layer: gram_matrix(style_features2[layer]) for layer in style_features2}

generated_img = content_img.clone().requires_grad_(True)

criterion = nn.MSELoss()
optimizer = optim.LBFGS([generated_img], max_iter=20)

style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}

content_weight = 1
w1 = 1e6  
w2 = 1e5  
tv_weight = 1e-6

def total_variation_loss(image):
    x_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
    return torch.mean(torch.abs(x_diff)) + torch.mean(torch.abs(y_diff))

epochs = 300

for epoch in range(1, epochs + 1):
    def closure():
        optimizer.zero_grad()
        gen_features = get_features(generated_img, vgg)
        
        content_loss = content_weight * criterion(gen_features['conv4_2'], content_features['conv4_2'])
        
        style_loss1 = 0
        for layer in style_weights:
            gen_gram = gram_matrix(gen_features[layer])
            style_loss1 += style_weights[layer] * criterion(gen_gram, style_grams1[layer])
        style_loss1 = w1 * style_loss1

        style_loss2 = 0
        for layer in style_weights:
            gen_gram = gram_matrix(gen_features[layer])
            style_loss2 += style_weights[layer] * criterion(gen_gram, style_grams2[layer])
        style_loss2 = w2 * style_loss2
        
        tv_loss = tv_weight * total_variation_loss(generated_img)
        
        total_loss = content_loss + style_loss1 + style_loss2 + tv_loss
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs}")
        plt.imshow(im_convert(generated_img))
        plt.axis('off')
        plt.savefig(f'___output3_epoch_{epoch}.jpg')

final_img = im_convert(generated_img)
plt.imsave('output_multi_style3.jpg', final_img)