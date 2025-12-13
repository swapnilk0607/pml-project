# enhanced_feature_extraction.ipynb

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, corner_harris, corner_peaks
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2hsv
import pandas as pd
import os

class EnhancedCricketFeatureExtractor:
    """
    Enhanced feature extraction with domain-specific cricket features.
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
    
    def extract_hsv_color_features(self, tile_rgb):
        """
        Extract HSV-based color features (better for ball/field separation).
        """
        tile_hsv = rgb2hsv(tile_rgb)
        
        # HSV histograms (more discriminative than RGB for colored objects)
        hist_h, _ = np.histogram(tile_hsv[:, :, 0], bins=16, range=(0, 1))
        hist_s, _ = np.histogram(tile_hsv[:, :, 1], bins=16, range=(0, 1))
        hist_v, _ = np.histogram(tile_hsv[:, :, 2], bins=16, range=(0, 1))
        
        hsv_hist = np.concatenate([hist_h, hist_s, hist_v]).astype('float')
        hsv_hist /= (hsv_hist.sum() + 1e-7)
        
        # Color moments (mean, std, skewness) for each channel
        color_moments = []
        for i in range(3):
            channel = tile_hsv[:, :, i].ravel()
            color_moments.extend([
                np.mean(channel),
                np.std(channel),
                np.mean((channel - np.mean(channel))**3)  # skewness
            ])
        
        return np.concatenate([hsv_hist, color_moments])
    
    def extract_edge_density_features(self, tile_gray):
        """
        Extract edge density features (important for bat/stump detection).
        """
        # Canny edge detection
        tile_uint8 = (tile_gray * 255).astype(np.uint8)
        edges = cv2.Canny(tile_uint8, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        
        # Edge direction histogram
        sobelx = cv2.Sobel(tile_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(tile_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edge_angles = np.arctan2(sobely, sobelx)
        edge_hist, _ = np.histogram(edge_angles, bins=8, range=(-np.pi, np.pi))
        edge_hist = edge_hist.astype('float') / (np.sum(edge_hist) + 1e-7)
        
        return np.concatenate([[edge_density], edge_hist])
    
    def extract_shape_features(self, tile_gray):
        """
        Extract shape-based features (circularity for ball detection).
        """
        tile_uint8 = (tile_gray * 255).astype(np.uint8)
        
        # Thresholding
        _, thresh = cv2.threshold(tile_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contour features
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-7)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / (h + 1e-7)
            
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-7)
            
            return np.array([len(contours), area/(thresh.size + 1e-7), 
                           circularity, aspect_ratio, solidity])
        else:
            return np.zeros(5)
    
    def extract_texture_features_advanced(self, tile_gray):
        """
        Enhanced texture features using multiple LBP radii.
        """
        tile_uint8 = (tile_gray * 255).astype(np.uint8)
        
        all_lbp_features = []
        
        # Multi-scale LBP
        for P, R in [(8, 1), (16, 2)]:
            lbp = local_binary_pattern(tile_uint8, P=P, R=R, method='uniform')
            n_bins = P + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            hist = hist.astype('float') / (hist.sum() + 1e-7)
            all_lbp_features.append(hist)
        
        return np.concatenate(all_lbp_features)
    
    def extract_statistical_features(self, tile_gray):
        """
        Statistical features of pixel intensities.
        """
        pixels = tile_gray.ravel()
        return np.array([
            np.mean(pixels),
            np.std(pixels),
            np.median(pixels),
            np.percentile(pixels, 25),
            np.percentile(pixels, 75),
            np.min(pixels),
            np.max(pixels)
        ])
    
    def extract_gabor_features(self, tile_gray):
        """
        Gabor filter features for texture and orientation.
        """
        tile_uint8 = (tile_gray * 255).astype(np.uint8)
        gabor_features = []
        
        # Gabor kernels at different orientations and frequencies
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for frequency in [0.1, 0.2]:
                kernel = cv2.getGaborKernel((21, 21), 5, theta, 10/frequency, 0.5, 0)
                filtered = cv2.filter2D(tile_uint8, cv2.CV_64F, kernel)
                gabor_features.extend([np.mean(filtered), np.std(filtered)])
        
        return np.array(gabor_features)
    
    def extract_all_features_from_tile(self, tile_gray, tile_rgb):
        """
        Extract comprehensive feature set from a single tile.
        """
        features = {}
        
        # 1. HOG features (gradient-based)
        hog_feat = hog(tile_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        features['hog'] = hog_feat
        
        # 2. Multi-scale LBP (texture)
        features['lbp'] = self.extract_texture_features_advanced(tile_gray)
        
        # 3. HSV color features (better for cricket objects)
        features['hsv_color'] = self.extract_hsv_color_features(tile_rgb)
        
        # 4. Edge density (for bat/stump edges)
        features['edge'] = self.extract_edge_density_features(tile_gray)
        
        # 5. Shape features (for ball circularity)
        features['shape'] = self.extract_shape_features(tile_gray)
        
        # 6. Statistical features
        features['stats'] = self.extract_statistical_features(tile_gray)
        
        # 7. Gabor features (texture/orientation)
        features['gabor'] = self.extract_gabor_features(tile_gray)
        
        return features
    
    def extract_features_from_image(self, file):
        """
        Extract features from all tiles of an image.
        """
        file_path = os.path.join(self.data_dir, file)
        
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            return
        
        try:
            image = imread(file_path)
        except:
            print(f"Error reading {file}")
            return
        
        # Resize and prepare
        image_rgb = resize(image, (800, 600), anti_aliasing=True)
        gray_image = rgb2gray(image_rgb)
        
        # Get dimensions
        height, width = gray_image.shape
        tile_height = height // 8
        tile_width = width // 8
        
        print(f"Processing {file}...")
        
        # Extract features from each tile
        for i in range(8):
            for j in range(8):
                y_start = i * tile_height
                y_end = (i + 1) * tile_height
                x_start = j * tile_width
                x_end = (j + 1) * tile_width
                
                tile_gray = gray_image[y_start:y_end, x_start:x_end]
                tile_rgb = image_rgb[y_start:y_end, x_start:x_end]
                
                # Extract all features
                tile_features = self.extract_all_features_from_tile(tile_gray, tile_rgb)
                
                # Store each feature type
                for feat_type, feat_vec in tile_features.items():
                    self.data.append({
                        'image': file,
                        'tile_i': i,
                        'tile_j': j,
                        'type': feat_type,
                        'features': feat_vec
                    })
        
        print(f"✓ Completed {file}")
    
    def write_features_to_file(self, filename='enhanced_features.csv'):
        """
        Write extracted features to CSV file.
        """
        # Group by image and tile
        grouped_data = {}
        
        for item in self.data:
            key = (item['image'], item['tile_i'], item['tile_j'])
            if key not in grouped_data:
                grouped_data[key] = {
                    'image': item['image'],
                    'tile_i': item['tile_i'],
                    'tile_j': item['tile_j']
                }
            grouped_data[key][item['type']] = item['features']
        
        # Create records
        records = []
        for key, tile_data in grouped_data.items():
            record = {
                'image': tile_data['image'],
                'tile_i': tile_data['tile_i'],
                'tile_j': tile_data['tile_j']
            }
            
            # Add all feature types
            for feat_type in ['hog', 'lbp', 'hsv_color', 'edge', 'shape', 'stats', 'gabor']:
                if feat_type in tile_data:
                    for idx, val in enumerate(tile_data[feat_type]):
                        record[f'x_{feat_type}_{idx}'] = val
            
            records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        
        print(f'\n{"="*70}')
        print(f'✓ Saved {len(df)} records to {filename}')
        print(f'  Shape: {df.shape}')
        print(f'  Feature columns: {df.shape[1] - 3}')
        print(f'{"="*70}\n')
        
        return df


# Usage
# extractor = EnhancedCricketFeatureExtractor(data_dir='../../../notebooks/sushant07/raw_image/resized/wtc')

# for file in os.listdir(extractor.data_dir):
#     extractor.extract_features_from_image(file)

# df = extractor.write_features_to_file('enhanced_features_test_images_7.csv')


# ---- Module-level convenience wrappers ----
def extract_features_from_dir(data_dir: str, output_csv: str = './output/enhanced_features.csv',
                              valid_exts=(".png", ".jpg", ".jpeg", ".webp")) -> pd.DataFrame:
    """
    Extract features for all images in a directory and write to CSV.

    Parameters:
    - data_dir: Folder containing images.
    - output_csv: Path to write the features CSV file.
    - valid_exts: Allowed image extensions.

    Returns:
    - pandas.DataFrame of extracted features.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv) or '.'
    os.makedirs(out_dir, exist_ok=True)

    extractor = EnhancedCricketFeatureExtractor(data_dir=data_dir)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)]
    for f in files:
        extractor.extract_features_from_image(f)
    return extractor.write_features_to_file(output_csv)


def extract_features_for_images(image_paths, output_csv: str = None) -> pd.DataFrame:
    """
    Extract features for a provided list of image paths (possibly in different folders).

    If output_csv is provided, writes the aggregated features to that CSV and returns the DataFrame.

    Parameters:
    - image_paths: Iterable of absolute or relative image file paths.
    - output_csv: Optional CSV path to persist results. If None, results are returned only.

    Returns:
    - pandas.DataFrame of extracted features.
    """
    # Group paths by their parent directory to reuse the extractor efficiently
    grouped = {}
    for p in image_paths:
        dirpath = os.path.dirname(p) or "."
        grouped.setdefault(dirpath, []).append(os.path.basename(p))

    all_records = []
    extractor = None
    for dirpath, files in grouped.items():
        extractor = EnhancedCricketFeatureExtractor(data_dir=dirpath)
        for f in files:
            extractor.extract_features_from_image(f)

        # Convert in-memory records to a DataFrame without forcing a CSV per group
        # Use a temp file in ./output when aggregating groups
        temp_csv = output_csv or './output/temp_features.csv'
        os.makedirs(os.path.dirname(temp_csv) or '.', exist_ok=True)
        df_group = extractor.write_features_to_file(filename=temp_csv)
        all_records.append(df_group)

    if not all_records:
        return pd.DataFrame()

    df = pd.concat(all_records, ignore_index=True)

    # If caller asked for a CSV but we processed in chunks, write the final concat
    if output_csv:
        # Ensure output directory exists before saving final CSV
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df