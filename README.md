## How It Works
1. **Blur Detection**:
   - Reads the image in grayscale.
   - Calculates the Laplacian variance of the image.
   - If the variance is below the threshold, the image is considered blurry.
   - Adaptive thresholding adjusts the threshold based on the contrast of the image histogram.

2. **Similarity Detection**:
   - Extracts features from images using a pre-trained ResNet-18 model.
   - Compares features using cosine similarity.
   - If similarity exceeds the threshold, the image is marked as similar.

## Example Output
- Images classified as blurry are moved to `processed_images/blurred_images/`.
- Images classified as similar are moved to `processed_images/similar_images/`.

## Notes
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- Ensure that the input folder contains valid image files.

For testing you can use the folder images. Before that make sure to create a conda env using the yml file.
