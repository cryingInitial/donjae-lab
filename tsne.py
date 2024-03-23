import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from backbone import get_model
from data import get_statistics
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import ImageFolder


# load model
data_name = 'cifar10'
model = get_model(ckpt_path="./checkpoints/neggrad_cifar10_class_4_5000_10.0.pth").cuda()
mean, std, image_size, num_classes = get_statistics(data_name)
    
normalize = transforms.Normalize(mean=mean, std=std)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
index = []
for idx in range(10):
    index += list(range(idx*1000, idx*1000+500))

testset = ImageFolder(root=f'./dataset/{data_name}/test', transform=test_transform)
testset = Subset(testset, index)
test_dataloader = DataLoader(dataset=testset, batch_size=500)
features_dict = {}

model.eval()
with torch.no_grad():
    for (x, y) in tqdm(test_dataloader):
        x, y = x.cuda(), y.cuda()
        result, embeddings=model(x, get_embeddings=True)
        features_dict[y[0].item()] = result
        pred = torch.argmax(result, dim=1)


# 모든 클래스의 데이터와 레이블을 하나의 배열로 결합
all_features = torch.cat(list(features_dict.values()), dim=0).cpu().numpy()
all_labels = np.concatenate([
    np.full(features.shape[0], label)
    for label, features in features_dict.items()
])

# t-SNE를 사용하여 2차원으로 차원 축소
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(all_features)

# 결과 시각화
plt.figure(figsize=(10, 8))
for label in features_dict.keys():
    indices = all_labels == label
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {label}')
plt.legend()
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE visualization of features')
plt.show()
os.makedirs("T-SNE", exist_ok=True)
plt.savefig("T-SNE/neggrad_10.png")