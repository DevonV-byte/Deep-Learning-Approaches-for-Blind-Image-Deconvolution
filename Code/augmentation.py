import random
from PIL import Image, ImageEnhance
import numpy as np

def perform_augmentation(image_paths, augmentation_rate=1, grayscale=False):
    print("Performing augmentation on images...")
    
    augmented_images = []
    techniques = ["rotate", "flip", "brightness"]
    
    # Define the number of augmentations per image based on the rate
    augmentation_multiplier = {1: 3, 2: 5, 3: 10}
    num_augmentations = augmentation_multiplier.get(augmentation_rate, 2)

    for image_path in image_paths:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if specified
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Apply augmentations
        for _ in range(num_augmentations):
            augmented_image = img.copy()
            
            # Randomly apply augmentation techniques
            for technique in random.sample(techniques, k=len(techniques)):
                if technique == "rotate":
                    angle = random.choice([90, 180, 270])
                    augmented_image = augmented_image.rotate(angle)
                elif technique == "flip":
                    if random.choice(["horizontal", "vertical"]) == "horizontal":
                        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        augmented_image = augmented_image.transpose(Image.FLIP_TOP_BOTTOM)
                elif technique == "brightness":
                    enhancer = ImageEnhance.Brightness(augmented_image)
                    augmented_image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Convert to NumPy array for consistency
            augmented_images.append(np.array(augmented_image))
    
    print(f"Augmentation completed. Total augmented images: {len(augmented_images)}.")
    return augmented_images
