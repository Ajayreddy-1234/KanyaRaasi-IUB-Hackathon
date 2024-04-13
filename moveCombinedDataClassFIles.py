import os
import shutil

def filter_and_move_files(source_folder, target_folder, class_name):
    """
    Move image and text files to a new folder if the text file contains a specific class name.

    Args:
    - source_folder (str): The directory containing the original dataset.
    - target_folder (str): The directory where files should be moved if they meet the condition.
    - class_name (str): The class name to search for in the text files.
    """

    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Loop over files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(source_folder, filename)
            
            # Read the contents of the file
            with open(file_path, 'r') as file:
                contents = file.read()
                
                # Check if the class name is in the file
                if class_name in contents:
                    # Get the base name without extension
                    base_name = filename.replace('.txt', '')
                    
                    # Define paths for the image and text files
                    img_file = f"{base_name}.png"
                    txt_file = f"{base_name}.txt"
                    
                    # Define source paths
                    source_img_path = os.path.join(source_folder, img_file)
                    source_txt_path = os.path.join(source_folder, txt_file)
                    
                    # Define target paths
                    target_img_path = os.path.join(target_folder, img_file)
                    target_txt_path = os.path.join(target_folder, txt_file)
                    
                    # Move the files
                    shutil.move(source_img_path, target_img_path)
                    shutil.move(source_txt_path, target_txt_path)
                    print(f"Moved: {img_file} and {txt_file}")

# Example usage
source_folder = 'path/to/your/dataset'
target_folder = 'path/to/your/dataset/test'
class_name = 'combined_dataset'
filter_and_move_files(source_folder, target_folder, class_name)
