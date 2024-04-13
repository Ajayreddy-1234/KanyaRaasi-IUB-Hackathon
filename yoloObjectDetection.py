import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Data Preprocessing
# Example of class name to index mapping
class_name_to_index = {
    "battery": 0,
    "biological": 1,
    "cardboard": 2,
    "clothes": 3,
    "glass": 4,
    "metal": 5,
    "paper": 6,
    "plastic": 7,
    "shoes": 8,
    "trash": 9
}


class YOLODataset(Dataset):
    def __init__(self, folder_path, img_size=416,num_classes = 10, class_name_to_index=None):
        super(YOLODataset, self).__init__()
        self.data = []
        self.num_classes = num_classes
        self.img_size = img_size

        # Load data
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                label_path = img_path.replace('.jpg', '.txt')

                # Read and resize image
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                img = img.transpose(2, 0, 1)  # From HWC to CHW
                img = torch.from_numpy(img / 255.0).float()  # Normalize

                # Read labels
                labels = []
                with open(label_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        class_name, x_min, y_min, x_max, y_max = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        class_idx = class_name_to_index[class_name]
                        x_center = (x_min + x_max) / 2 / img.shape[1]
                        y_center = (y_min + y_max) / 2 / img.shape[2]
                        width = (x_max - x_min) / img.shape[1]
                        height = (y_max - y_min) / img.shape[2]
                         # One-hot encoding for class labels
                        class_label = torch.zeros(self.num_classes)
                        class_label[class_idx] = 1
                        labels.append(torch.cat((class_label, torch.tensor([x_center, y_center, width, height]))))

                self.data.append((img, torch.stack(labels)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Usage
folder_path = '../dataset/combined_dataset'
dataset = YOLODataset(folder_path, 416, 10, class_name_to_index)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# Model Architecture

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels // 2, in_channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))  # Skip connection

# Adding to the YOLOv4 model
class YOLOv4(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv4, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 3
        self.output_filters = self.num_anchors * (5 + num_classes)
        
        # Define a more complex YOLOv4-like architecture
        self.layer1 = ConvBlock(3, 32, 3, 1, 1)
        self.layer2 = ConvBlock(32, 64, 3, 2, 1)
        self.layer3 = ConvBlock(64, 128, 3, 2, 1)
        self.residual_block = ResidualBlock(128)  # Example of adding a residual block
        self.output_layer = nn.Conv2d(128, self.output_filters, 1, 1, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.residual_block(x)  # Using the residual block
        x = self.output_layer(x)
        return x.reshape(x.size(0), self.num_anchors, self.num_classes + 5, x.size(2), x.size(3)).permute(0, 3, 4, 1, 2)


# Training

import torch.optim as optim
import torch.nn.functional as F

def yolo_loss(outputs, targets, S=7, B=2, C=10):
    lambda_coord = 5
    lambda_noobj = 0.5
    lambda_obj = 1

    batch_size = outputs.shape[0]
    pred_boxes = outputs[:, :, :, :B*5].reshape(-1, S, S, B, 5)
    pred_classes = outputs[:, :, :, B*5:].reshape(-1, S, S, C)
    
    true_boxes = targets[:, :, :, :B*5].reshape(-1, S, S, B, 5)
    true_classes = targets[:, :, :, B*5:].reshape(-1, S, S, C)

    object_mask = true_boxes[..., 4] == 1
    no_object_mask = true_boxes[..., 4] == 0
    
    coord_loss = lambda_coord * torch.sum(object_mask * torch.sum((pred_boxes[..., :4] - true_boxes[..., :4])**2, dim=-1))
    obj_confidence_loss = lambda_obj * torch.sum(object_mask * (pred_boxes[..., 4] - true_boxes[..., 4])**2)
    noobj_confidence_loss = lambda_noobj * torch.sum(no_object_mask * (pred_boxes[..., 4]**2))
    
    # Need actual class indices for cross-entropy loss
    true_class_indices = torch.argmax(true_classes, dim=-1)  # Get class indices from one-hot
    class_loss = F.cross_entropy(pred_classes[object_mask], true_class_indices[object_mask], reduction='sum')

    total_loss = coord_loss + obj_confidence_loss + noobj_confidence_loss + class_loss
    return total_loss / batch_size


# Setup the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv4(num_classes=10).to(device)  # Adjust num_classes as necessary
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = yolo_loss  # You need to define this based on YOLOv4 specifics

def calculate_iou(pred_boxes, true_boxes):
    """
    Calculate intersection over union (IoU) between predicted and true boxes.

    Args:
    - pred_boxes (tensor): Predicted bounding boxes of shape [N, 4], where N is the number of boxes, and each box is x_center, y_center, width, height.
    - true_boxes (tensor): True bounding boxes of shape [N, 4].

    Returns:
    - iou (tensor): The IoU scores for each corresponding pair of boxes.
    """
    # Convert from center coordinates to bounding box corners.
    pred_corners = torch.zeros_like(pred_boxes)
    pred_corners[:, 0] = pred_boxes[:, 0] - pred_boxes[:, 2] / 2  # x_min
    pred_corners[:, 1] = pred_boxes[:, 1] - pred_boxes[:, 3] / 2  # y_min
    pred_corners[:, 2] = pred_boxes[:, 0] + pred_boxes[:, 2] / 2  # x_max
    pred_corners[:, 3] = pred_boxes[:, 1] + pred_boxes[:, 3] / 2  # y_max

    true_corners = torch.zeros_like(true_boxes)
    true_corners[:, 0] = true_boxes[:, 0] - true_boxes[:, 2] / 2  # x_min
    true_corners[:, 1] = true_boxes[:, 1] - true_boxes[:, 3] / 2  # y_min
    true_corners[:, 2] = true_boxes[:, 0] + true_boxes[:, 2] / 2  # x_max
    true_corners[:, 3] = true_boxes[:, 1] + true_boxes[:, 3] / 2  # y_max

    # Calculate the coordinates of the intersection rectangle
    inter_x_min = torch.max(pred_corners[:, 0], true_corners[:, 0])
    inter_y_min = torch.max(pred_corners[:, 1], true_corners[:, 1])
    inter_x_max = torch.min(pred_corners[:, 2], true_corners[:, 2])
    inter_y_max = torch.min(pred_corners[:, 3], true_corners[:, 3])

    # Intersection area
    inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)

    # Area of both rectangles
    pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
    true_area = (true_corners[:, 2] - true_corners[:, 0]) * (true_corners[:, 3] - true_corners[:, 1])

    # Union area
    union_area = pred_area + true_area - inter_area

    # IoU calculation
    iou = inter_area / union_area

    return iou

def calculate_accuracy(outputs, targets, iou_threshold=0.5):
    """
    Calculate detection accuracy based on IoU and class accuracy.

    Args:
    - outputs (tensor): Model outputs of the shape [batch_size, num_anchors, 5+num_classes, grid_size, grid_size]
    - targets (list of tensors): List of tensors with ground truth boxes and classes

    Returns:
    - accuracy (float): The average accuracy over the batch.
    """
    batch_size = outputs.shape[0]
    correct_predictions = 0

    for i in range(batch_size):
        pred_boxes = outputs[i, :, :4]  # Assuming outputs are already in xywh format
        pred_classes = torch.argmax(outputs[i, :, 5:], dim=-1)
        true_boxes = targets[i][:, 1:5]
        true_classes = targets[i][:, 0]

        # Calculate IoU for the predicted and true boxes
        ious = calculate_iou(pred_boxes, true_boxes)
        # Check if IoU > threshold and classes match
        class_matches = (pred_classes == true_classes).float()
        correct_predictions += torch.sum((ious >= iou_threshold) & class_matches)

    accuracy = correct_predictions / batch_size
    return accuracy.item()  # Convert to Python float for easier handling

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for images, batch_targets in train_loader:
        images = images.to(device)
        # Ensure targets are correctly formatted for the loss function:
        targets = torch.cat([t.to(device) for t in batch_targets], dim=0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)  # Ensure the loss function accepts this format
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, targets)  # Ensure this function is appropriate
        
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    print(f'Epoch {epoch+1}, Training Loss: {avg_loss}, Training Accuracy: {avg_accuracy}')

    # Validation loop every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, batch_targets in val_loader:
                images = images.to(device)
                targets = torch.cat([t.to(device) for t in batch_targets], dim=0)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                val_accuracy += calculate_accuracy(outputs, targets)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')

print("Training complete.")
