import os
import cv2
import shutil
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity

# --- Enhanced Blur Detection ---
def detect_blur(image_path, threshold=100.0, adaptive=False):
    """
    Detect if an image is blurry using the Laplacian Variance method with optional adaptive thresholding.
    :param image_path: Path to the image.
    :param threshold: Variance threshold below which the image is considered blurry.
    :param adaptive: Whether to use adaptive thresholding based on histogram analysis.
    :return: True if the image is blurry, False otherwise, along with the blur score.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, 0.0

    # Compute Laplacian variance
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # Adaptive thresholding based on histogram analysis
    if adaptive:
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        contrast_score = np.std(hist)  # Measure of contrast
        dynamic_threshold = threshold * (contrast_score / 50)  # Scale threshold based on contrast
        return laplacian_var < dynamic_threshold, laplacian_var

    return laplacian_var < threshold, laplacian_var

# --- Main Program ---
def process_folder(input_folder, output_folder, blur_threshold=100.0, similarity_threshold=0.9):
    """
    Process a folder to detect blurred and similar images, and move them to appropriate output folders.
    :param input_folder: Path to the input folder containing images.
    :param output_folder: Path to the output folder to store processed images.
    :param blur_threshold: Variance threshold for blur detection.
    :param similarity_threshold: Threshold to classify images as similar.
    """
    os.makedirs(output_folder, exist_ok=True)
    blurred_folder = os.path.join(output_folder, "blurred_images")
    similar_folder = os.path.join(output_folder, "similar_images")
    os.makedirs(blurred_folder, exist_ok=True)
    os.makedirs(similar_folder, exist_ok=True)

    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []

    # Collect image paths
    for root, _, files in os.walk(input_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.splitext(filename)[1].lower() in supported_formats:
                image_paths.append(file_path)

    print(f"Found {len(image_paths)} valid images in {input_folder}")

    # Detect blurred images
    non_blurred_images = []
    for image_path in image_paths:
        is_blurry, score = detect_blur(image_path, blur_threshold, adaptive=True)
        if is_blurry:
            print(f"Blurry image: {os.path.basename(image_path)} (Score: {score:.2f})")
            shutil.move(image_path, os.path.join(blurred_folder, os.path.basename(image_path)))
        else:
            non_blurred_images.append(image_path)

    # Load pre-trained ResNet model with updated API
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()  # Remove the classification layer
    resnet.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Detect similar images
    features = {}
    visited = set()
    for image_path in non_blurred_images:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features[image_path] = resnet(image_tensor).squeeze(0).numpy()

    for i, image_path in enumerate(non_blurred_images):
        if image_path in visited:
            continue
        for j in range(i + 1, len(non_blurred_images)):
            if non_blurred_images[j] in visited:
                continue
            similarity = cosine_similarity(
                [features[image_path]], [features[non_blurred_images[j]]]
            )[0][0]
            if similarity > similarity_threshold:
                print(f"Similar images: {os.path.basename(image_path)} and {os.path.basename(non_blurred_images[j])}")
                shutil.move(non_blurred_images[j], os.path.join(similar_folder, os.path.basename(non_blurred_images[j])))
                visited.add(non_blurred_images[j])

    print(f"Blurred images moved to {blurred_folder}")
    print(f"Similar images moved to {similar_folder}")

if __name__ == "__main__":
    # Input and output folders
    input_folder = input("Enter the path to the folder containing images: ").strip()
    output_folder = "processed_images"

    process_folder(input_folder, output_folder)
