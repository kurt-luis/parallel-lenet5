import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

def convert_jpg_to_bin(image_path, output_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"Loading image: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    tensor_img = transform(img)
    flat_numpy_array = tensor_img.numpy().flatten().astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flat_numpy_array.tofile(output_path)
    
    print(f"Success! Saved binary to {output_path}")
    print(f"Total floats: {flat_numpy_array.size} (Expected: 3x32x32 = 3072)")

if __name__ == "__main__":
    input_jpg = "../data/processed/car_cifar.jpg" 
    output_bin = "../data/processed/test_image_0.bin" 
    
    convert_jpg_to_bin(input_jpg, output_bin)