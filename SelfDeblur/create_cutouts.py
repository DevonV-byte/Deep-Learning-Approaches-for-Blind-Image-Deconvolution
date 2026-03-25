from PIL import Image
import os

def create_cutouts(image_path, output_dir, cutout_size=(255, 255)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image
    img = Image.open(image_path)
    width, height = img.size
    
    # Calculate positions for cutouts
    center_x = width // 2
    center_y = height // 2
    half_size_x = cutout_size[0] // 2
    half_size_y = cutout_size[1] // 2
    
    # Define cutout positions with sequential numbering
    positions = {
        'im1': (center_x - half_size_x, center_y - half_size_y),  # center
        'im2': (0, 0),                                            # top left
        'im3': (width - cutout_size[0], 0),                       # top right
        'im4': (width - cutout_size[0], center_y - half_size_y),  # center right
        'im5': (center_x - half_size_x, height - cutout_size[1])  # bottom center
    }
    
    # Create and save cutouts
    for position_name, (x, y) in positions.items():
        cutout = img.crop((x, y, x + cutout_size[0], y + cutout_size[1]))
        output_path = os.path.join(output_dir, f'cutout_{position_name}.png')
        cutout.save(output_path)
        print(f"Saved {position_name} cutout to {output_path}")

if __name__ == "__main__":
    # Define paths
    input_image = "Datasets/Hubble/Base/potw2050a.tif"
    output_directory = "Datasets/Hubble/Cutouts"
    
    # Create cutouts
    create_cutouts(input_image, output_directory) 