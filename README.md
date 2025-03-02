# KanyaRaasi Hackathon Project

## About

This project was developed as a submission for the IUB Hackathon. It features modules for data augmentation, model training, and object detection using YOLO. The aim is to deliver a proof-of-concept solution that leverages computer vision and deep learning techniques to address real-world challenges.

## Features

- **Data Augmentation**: Enhance your dataset using various augmentation techniques. See `augment.py` for details.
- **Model Training**: Build and train machine learning models with the provided scripts (`model.py`).
- **Object Detection**: Implement YOLO-based object detection for image processing tasks via `yoloObjectDetection.py`.
- **Utility Functions**: Organize and manage combined data class files using `moveCombinedDataClassFIles.py`.

## Technology Stack

- **Programming Language**: Python
- **Deep Learning & Computer Vision**: YOLO for object detection and related frameworks
- **Data Processing**: Custom Python scripts for data augmentation and model training
- **Notebooks**: Jupyter Notebook files (if any) for exploratory analysis and evaluation

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt` for a full list)

### Installation

1. **Clone the Repository:**
 ```bash
 git clone https://github.com/Ajayreddy-1234/KanyaRaasi-IUB-Hackathon.git
 cd KanyaRaasi-IUB-Hackathon
```
2. **Set Up a Virtual Environment:**
```bash
python -m venv venv
```
3. **Activate the Virtual Environment:**
```bash
venv\Scripts\activate
```
4. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Project
1. Data Augmentation:
 Run the augmentation script:
 ```bash
python augment.py
```
2. Model Training:
Execute the model training script:
```bash
python model.py
```
3. Object Detection:
Run the YOLO object detection module:
```bash
python yoloObjectDetection.py
```

## Application Structure

- **augment.py**: Script for augmenting your dataset.
- **model.py**: Script for building and training the machine learning model.
- **moveCombinedDataClassFIles.py**: Utility script for organizing or moving combined data class files.
- **yoloObjectDetection.py**: Module for executing YOLO-based object detection.
- **Utilities/**: Additional utility scripts and resources.
- **requirements.txt**: List of required Python packages.
- **README.md**: This file.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests with improvements or bug fixes.

## License

This project is open source and available under the [MIT License](LICENSE).
