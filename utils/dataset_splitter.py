import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(src_folder, dest_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all categories (folders) in the source folder
    categories = os.listdir(src_folder)

    for category in categories:
        category_path = os.path.join(src_folder, category)
        
        if os.path.isdir(category_path):  # Ensure we're working with a folder
            # Create subdirectories for train, val, and test if they don't exist
            for split in ['train', 'val', 'test']:
                split_folder = os.path.join(dest_folder, split, category)
                if not os.path.exists(split_folder):
                    os.makedirs(split_folder)

            # Get all images in the category folder
            images = os.listdir(category_path)
            train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

            # Copy images to train, val, and test directories
            for split_images, split in zip([train_images, val_images, test_images], ['train', 'val', 'test']):
                for image in split_images:
                    src_image = os.path.join(category_path, image)
                    dest_image = os.path.join(dest_folder, split, category, image)
                    shutil.copy(src_image, dest_image)

    print("Dataset split and copied successfully!")
