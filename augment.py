import os
import random
from PIL import Image, ImageEnhance

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        with Image.open(img_path) as img:
            img = img.convert("RGBA")  # Ensure all images are in RGBA for uniform handling of transparency
            images.append((img, filename, folder.split(os.sep)[-1]))  # Include filename and class label
    return images

def check_overlap(new_box, boxes):
    for box in boxes:
        if (new_box[0] < box[2] and new_box[2] > box[0] and
            new_box[1] < box[3] and new_box[3] > box[1]):
            return True
    return False

def apply_transformations(img):
    # Rotate image
    angle = random.randint(-45, 45)
    img = img.rotate(angle, expand=True)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.7, 1.3)  # Change brightness randomly
    img = enhancer.enhance(factor)

    return img

def combine_images(image_list, background_size, min_items):
    background = Image.new('RGBA', background_size, (255, 255, 255, 255))
    annotations = []
    occupied_areas = []

    while len(image_list) < min_items:
        image_list += random.choices(image_list, k=min_items - len(image_list))

    for img, filename, label in image_list:
        img = apply_transformations(img)

        scale_x = background_size[0] / img.size[0]
        scale_y = background_size[1] / img.size[1]
        scale_factor = min(scale_x, scale_y, 1)  # Avoid upscaling
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size)

        placed = False
        attempts = 0
        while not placed and attempts < 50:
            max_x = max(background_size[0] - img.size[0], 0)
            max_y = max(background_size[1] - img.size[1], 0)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            new_box = [x, y, x + img.size[0], y + img.size[1]]
            if not check_overlap(new_box, occupied_areas):
                background.paste(img, (x, y), img)
                occupied_areas.append(new_box)
                annotations.append({
                    'class': label,
                    'bbox': new_box,
                    'filename': filename
                })
                placed = True
            attempts += 1

    return background.convert('RGB'), annotations

def create_dataset(base_folder, num_images=200, min_items_per_image=4, max_items_per_image=10, max_size=(2048, 2048)):
    categories = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    output_folder = os.path.join(base_folder, 'combined_dataset')
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        num_items = random.randint(min_items_per_image, max_items_per_image)
        background_size = (random.randint(500, max_size[0]), random.randint(500, max_size[1]))
        selected_images = []
        for _ in range(num_items):
            category_folder = random.choice(categories)
            images = load_images_from_folder(category_folder)
            if images:
                selected_image = random.choice(images)
                selected_images.append(selected_image)
        combined_image, annotations = combine_images(selected_images, background_size, min_items_per_image)
        combined_image.save(os.path.join(output_folder, f'combined_{i}.jpg'))

        # Save annotations
        with open(os.path.join(output_folder, f'combined_{i}.txt'), 'w') as f:
            for annotation in annotations:
                f.write(f"{annotation['class']} {annotation['bbox'][0]} {annotation['bbox'][1]} {annotation['bbox'][2]} {annotation['bbox'][3]}\n")

# Example usage:
create_dataset('archive')