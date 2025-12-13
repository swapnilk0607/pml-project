import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog, local_binary_pattern
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def load():
    # Initialize module-level globals so other functions can access them
    global data, data_dir, le
    data = []
    # data_dir = '../dataset/train_data'
    data_dir = 'input_images/'
    le = LabelEncoder()
    return data_dir

# Tile image to hog features extraction
def extract_hog_features(file):
    file_path = os.path.join(data_dir, file)
    image = imread(file_path)
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
    
    # Get image dimensions
    height, width = resized_image.shape
    
    # Calculate tile size
    tile_height = height // 8
    tile_width = width // 8
    
    # Create figure for 8x8 subplot
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle(f'Tiles for {file}', fontsize=16)
    # Dictionary to store features for this image
    hog_features = {}
    # Split image into 8x8 tiles
    for i in range(8):
        for j in range(8):
            # Calculate tile boundaries
            y_start = i * tile_height
            y_end = (i + 1) * tile_height
            x_start = j * tile_width
            x_end = (j + 1) * tile_width
            
            # Extract tile
            tile = resized_image[y_start:y_end, x_start:x_end]
            
            # Display tile in subplot
            axes[i, j].imshow(tile, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'{i},{j}', fontsize=6)
            
            # Extract HOG features from tile
            features = hog(tile, orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        block_norm='L2-Hys')
            # Store in dictionary with key: (image_name, i, j)
            key = (file, i, j)
            hog_features[key] = features
            
            # Also append to global data list
            data.append({
                'type': 'hog',
                'image': file,
                'tile_i': i,
                'tile_j': j,
                'features': features
            })
    #plt.tight_layout()
    #plt.show()
    return hog_features

def extract_lbp_features(file):
    file_path = os.path.join(data_dir, file)
    image = imread(file_path)
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
    # Get image dimensions
    height, width = resized_image.shape
    # Calculate tile size
    tile_height = height // 8
    tile_width = width // 8
    # Dictionary to store features for this image
    lbp_features = {}
    # Split image into 8x8 tiles
    for i in range(8):
        for j in range(8):
            # Calculate tile boundaries
            y_start = i * tile_height
            y_end = (i + 1) * tile_height
            x_start = j * tile_width
            x_end = (j + 1) * tile_width
            
            # Extract tile
            tile = resized_image[y_start:y_end, x_start:x_end]
            # Extract LBP features from tile
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(tile, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, n_points + 3),
                                     range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            
            # Store in dictionary with key: (image_name, i, j)
            key = (file, i, j)
            lbp_features[key] = hist
            
            # Also append to global data list
            data.append({
                'type': 'lbp',
                'image': file,
                'tile_i': i,
                'tile_j': j,
                'features': hist
            })
    # plt.tight_layout()
    # plt.show()
    return lbp_features

def write_features_to_file(filename='input_features/input.csv'):
    # Group data by (image, tile_i, tile_j)
    grouped_data = {}
    
    for item in data:
        key = (item['image'], item['tile_i'], item['tile_j'])
        if key not in grouped_data:
            grouped_data[key] = {
                'image': item['image'],
                'tile_i': item['tile_i'],
                'tile_j': item['tile_j'],
                'hog': None,
                'lbp': None
            }
        
        # Store features by type
        grouped_data[key][item['type']] = item['features']
    
    # Create records with combined features
    records = []
    for key, tile_data in grouped_data.items():
        record = {
            'image': tile_data['image'],
            'tile_i': tile_data['tile_i'],
            'tile_j': tile_data['tile_j']
        }
        
        # Add HOG features
        if tile_data['hog'] is not None:
            for idx, val in enumerate(tile_data['hog']):
                record[f'x_hog_{idx}'] = val
        
        # Add LBP features
        if tile_data['lbp'] is not None:
            for idx, val in enumerate(tile_data['lbp']):
                record[f'x_lbp_{idx}'] = val
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f'Saved {len(df)} records to {filename}')
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns[:10].tolist()}...')  # Show first 10 columns

def runExtraction():
    data_dir = load()
    # Run the extraction
    for file in os.listdir(data_dir):
      # Skip if it's a directory or not an image file
      if os.path.isdir(os.path.join(data_dir, file)) or not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
         continue
      hog_features = extract_hog_features(file)
      print(f'Extracted HOG features for {file}, total tiles: {len(hog_features)} and size of each tile feature vector: {len(next(iter(hog_features.values())))}')
      lbp_features = extract_lbp_features(file)
      print(f'Extracted LBP features for {file}, total tiles: {len(lbp_features)} and size of each tile feature vector: {len(next(iter(lbp_features.values())))}')
      #color_histogram_features = extract_color_histogram_features(file)
      #print(f'Extracted Color Histogram features for {file}, total tiles: {len(color_histogram_features)} and size of each tile feature vector: {len(next(iter(color_histogram_features.values())))}')
      write_features_to_file()

if __name__ == "__main__":
    runExtraction()
