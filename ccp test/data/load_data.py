import os
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch

# 定义数据变换，包括标准化
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集类
class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 加载NORMAL图像及其灰度视图
        normal_dir = os.path.join(root_dir, 'NORMAL')
        for file_name in os.listdir(normal_dir):
            if file_name.endswith('.jpeg'):
                file_path = os.path.join(normal_dir, file_name)
                self.images.append((file_path, 'gray'))
                self.labels.append(0)

        # 加载PNEUMONIA图像及其灰度视图
        pneumonia_dir = os.path.join(root_dir, 'PNEUMONIA')
        for file_name in os.listdir(pneumonia_dir):
            if file_name.endswith('.jpeg'):
                file_path = os.path.join(pneumonia_dir, file_name)
                self.images.append((file_path, 'gray'))
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        view1_path, view_type = self.images[idx]
        label = self.labels[idx]

        # 加载图像
        view1 = Image.open(view1_path).convert('RGB')
        
        # 转换为灰度图像
        if view_type == 'gray':
            view2 = Image.open(view1_path).convert('L').convert('RGB')
        else:
            view2 = view1

        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return view1, view2, label

# 加载预训练模型用于特征提取
def load_feature_extractor():
    model = models.resnet18(pretrained=True)
    # 移除最后的全连接层
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

# 提取特征向量和特征矩阵
def extract_features(model, dataloader):
    feature_vectors = []
    feature_matrices = []
    labels = []
    with torch.no_grad():
        for view1, view2, label in dataloader:
            # 提取每个视图的特征
            feature1 = model(view1).squeeze()
            feature2 = model(view2).squeeze()
            
            # 将特征向量和矩阵添加到列表
            feature_vectors.append((feature1.mean(dim=(1, 2)), feature2.mean(dim=(1, 2))))
            feature_matrices.append((feature1, feature2))
            labels.append(label)
    return feature_vectors, feature_matrices, labels

# 加载数据
def load_data(data_dir, batch_size=128):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    datasets = {
        'train': MultiViewDataset(train_dir, transform=transform),
        'val': MultiViewDataset(val_dir, transform=transform),
        'test': MultiViewDataset(test_dir, transform=transform)
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=True)
    }

    return dataloaders

# 使用示例
data_dir = '/path/to/chest_xray'
dataloaders = load_data(data_dir)
feature_extractor = load_feature_extractor()

# 提取特征
train_feature_vectors, train_feature_matrices, train_labels = extract_features(feature_extractor, dataloaders['train'])

# 验证特征提取
for (vector1, vector2), (matrix1, matrix2), labels in zip(train_feature_vectors, train_feature_matrices, train_labels):
    print("Vector Shapes:", vector1.shape, vector2.shape)
    print("Matrix Shapes:", matrix1.shape, matrix2.shape)
    break
