import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None, class_to_idx=None):
        self.root = root
        self.transforms = transforms
        self.class_to_idx = class_to_idx
        assert self.class_to_idx is not None, "class_to_idx mapping cannot be None"
        
        # Read all files and separate images from labels based on file extensions
        self.imgs = []
        self.labels = {}
        files = os.listdir(root)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # Add or change the conditions based on your file types
                self.imgs.append(file)
                # Assuming label file matches image file name but has .txt extension
                label_file = file.rsplit('.', 1)[0] + '.txt'
                if label_file in files:
                    self.labels[file] = label_file

        assert len(self.imgs) > 0, "No images found in the dataset directory"

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        label_path = os.path.join(self.root, self.labels[self.imgs[idx]])
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        classes = []
        try:
            with open(label_path) as f:
                for line in f:
                    class_name, xmin, ymin, xmax, ymax = line.split()
                    class_id = self.class_to_idx[class_name]
                    boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                    classes.append(class_id)
        except FileNotFoundError:
            print(f"Label file not found for {img_path}")
            return None
        except KeyError:
            print(f"Class name {class_name} not found in class_to_idx mapping")
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
          img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Example usage
class_to_idx = {'battery': 0, 'biological': 1, 'cardboard': 3, 'clothes': 4, 'glass': 5, 'metal': 6, 'paper': 7, 'plastic': 8, 'shoes': 9, 'trash': 10}  # Define your mapping here based on the dataset

from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

from torchvision.transforms import functional as F

class ComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform():
    transforms = [ToTensor()]
    return ComposeTransforms(transforms)

def get_model(num_classes):
    # Load a pre-trained model for fine-tuning
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    images = [item[0] for item in batch]  # Collect all images
    targets = [item[1] for item in batch]  # Collect all corresponding targets

    # Stack images as they should have the same dimensions after transformation
    images = torch.stack(images, 0)

    # Targets do not need to be stacked since they can have variable lengths
    return images, targets

# Create the dataset with transformations
dataset = CustomDataset(root='/content/drive/MyDrive/combined_dataset', transforms=get_transform(), class_to_idx=class_to_idx)
# Use this custom collate function in your DataLoader
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# import torch
# from torch.utils.data import DataLoader, SubsetRandomSampler

# import torch
# from torch.utils.data import DataLoader, SubsetRandomSampler

# # # Assuming 'dataset' is already created with CustomDataset
# num_samples = 20
# indices = torch.randperm(len(dataset))[:num_samples]

# sampler = SubsetRandomSampler(indices)
# train_loader = DataLoader(dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn, num_workers=1)

# # Now you can use this small_data_loader to check if your training loop starts and processes correctly
# for images, targets in small_data_loader:
#     print(images.shape, len(targets))
#     break  # Only take the first batch and break

# Initialize the model
model = get_model(num_classes=len(class_to_idx) + 1)  # Include background as a class

# Move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Choose the right optimizer and learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10  # Reduce epochs for quick testing
try:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        for images, targets in data_loader:
            start_time = time.time()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            losses.backward()
            optimizer.step()
            print(f"Batch processed in {time.time() - start_time:.2f} seconds")

        # Average the loss by the total number of batches
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch} finished - Loss: {avg_loss:.4f}, Duration: {time.time() - epoch_start_time:.2f} seconds")
except Exception as e:
    print(f"An error occurred: {e}")

# Save the model
torch.save(model.state_dict(), 'model.pth')

test_dataset = CustomDataset(root='../dataset/test', transforms=get_transform(), class_to_idx=class_to_idx)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_classes=len(class_to_idx) + 1)  # Include background as a class
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

from collections import defaultdict

def calculate_metrics(target_boxes, target_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """
    Calculate metrics such as IoU, precision, and recall for each class.
    """
    metrics = defaultdict(list)
    for i, label in enumerate(pred_labels):
        # Calculate IoU for each predicted box
        iou = calculate_iou(target_boxes, pred_boxes[i].unsqueeze(0))

        # Determine if the prediction is correct based on IoU and class
        correct = iou >= iou_threshold and label == target_labels[iou.argmax()]
        metrics[label.item()].append((iou.max().item(), correct))

    return metrics

def calculate_iou(target_boxes, pred_boxes):
    """
    Calculate intersection over union (IoU) between target boxes and predicted boxes.
    """
    inter_xmin = torch.max(target_boxes[:, 0], pred_boxes[:, 0])
    inter_ymin = torch.max(target_boxes[:, 1], pred_boxes[:, 1])
    inter_xmax = torch.min(target_boxes[:, 2], pred_boxes[:, 2])
    inter_ymax = torch.min(target_boxes[:, 3], pred_boxes[:, 3])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union_area = target_area + pred_area - inter_area

    return inter_area / union_area

def evaluate(model, data_loader, device):
    model.eval()
    all_metrics = defaultdict(list)

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i, output in enumerate(outputs):
            pred_boxes = output['boxes'].data
            pred_labels = output['labels'].data
            pred_scores = output['scores'].data

            target_boxes = targets[i]['boxes'].to(device)
            target_labels = targets[i]['labels'].to(device)

            # Filter out predictions with low confidence
            mask = pred_scores >= 0.5  # Confidence threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]

            # Calculate metrics
            metrics = calculate_metrics(target_boxes, target_labels, pred_boxes, pred_labels, pred_scores)
            for key, val in metrics.items():
                all_metrics[key].extend(val)

    # Calculate and print average IoU and class accuracy
    for class_id, data in all_metrics.items():
        ious = [x[0] for x in data]
        correct_predictions = [x[1] for x in data]
        avg_iou = sum(ious) / len(ious) if ious else 0
        accuracy = sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0
        print(f'Class {class_id}: Average IoU: {avg_iou:.4f}, Accuracy: {accuracy:.4f}')

# Assuming model and test_loader have been initialized
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluate(model, test_loader, device)