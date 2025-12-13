import os
import pickle
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
# ML library
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import pickle
import re
from feature_extraction import runExtraction
def run_detection_pipeline(image_dir):
    """
    Reads images from a directory and performs object detection.
    Returns a list of dictionaries containing filename, detections, and metadata.
    
    INTEGRATION INSTRUCTIONS:
    ===========================
    1. Place your pickle model file (e.g., 'model.pkl') in the app directory
    2. Uncomment the model loading code below
    3. Replace the mock detection logic with your model's prediction code
    4. Ensure output format matches the expected structure
    
    SKLEARN COMPATIBILITY FIX:
    If you get "ValueError: node array from the pickle has an incompatible dtype":
    - Run: pip install --upgrade scikit-learn
    - Or: pip install scikit-learn==1.3.2 (matching pickle creation version)
    """
    
    # Get all images from the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', 'webp'))]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return []
    
    results = []
    
    # Classes typical for cricket
    classes = ['Batsman', 'Bowler', 'Ball', 'Stump', 'Umpire']
    
    print(f"Processing {len(image_files)} images from {image_dir}...")
    
    try:
        with open('RandomForestClassifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("ERROR: Model file 'RandomForestClassifier_model.pkl' not found!")
        print("Please place your model pickle file in the app directory.")
        return []
    except ValueError as e:
        if "incompatible dtype" in str(e):
            print("ERROR: scikit-learn version mismatch!")
            print("Solution: pip install --upgrade scikit-learn")
            print(f"Details: {str(e)}")
        return []
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return []
    # ===== END MODEL LOADING SECTION =====
    
    for img_file in image_files:
        try:
            # Simulate processing time (remove when using real model)
            time.sleep(0.3) 
            
            # ===== REPLACE THIS WITH YOUR MODEL INFERENCE =====
            # Example with scikit-learn model:
            # 
            # from PIL import Image
            # import numpy as np
            # 
            # image = Image.open(os.path.join(image_dir, img_file))
            # # Preprocess image as needed
            # image_array = np.array(image)
            # 
            # # Get predictions from your model
            # predictions = model.predict([image_array])
            # 
            # # Convert predictions to detection format
            # detections = format_detections(predictions)
            # ===== END MODEL INFERENCE SECTION =====
            
            # file_path = '../outputs/features_data_grp_1_master.csv'
            # file_path = '../outputs/features_data_grp_1_master.csv'
            runExtraction()
            file_path = 'input_features/input.csv'
            np.random.seed(42)

            df = pd.read_csv(file_path)
            df_clean = df.dropna()
            pattern = r'^x_ch.*'
            columns_to_drop = [col for col in df_clean.columns if re.match(pattern, col)]
            df_clean = df_clean.drop(columns=columns_to_drop)

            X = df_clean.drop(['image', 'tile_i', 'tile_j'], axis=1)

            #Apply MinMaxScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            #Load PCA pickle file
            # with open('../models/pca_model.pkl', 'rb') as f:
            #     pca = pickle.load(f)
            # X_pca = pca.transform(X_scaled)

       

            #Predict using the loaded model
            y_pred = model.predict(X_scaled)
            print(y_pred)

            # Mock detection data (Remove this when using real model)
            detections = []
            num_objects = random.randint(3, 10)
            
            for _ in range(num_objects):
                detections.append({
                    'class': random.choice(classes),
                    'confidence': random.uniform(0.70, 0.99),
                    'bbox': [random.randint(0, 50), random.randint(0, 50), 100, 200] 
                })
            
            results.append({
                'filename': img_file,
                'detections': detections,
                'processed_at': time.strftime("%H:%M:%S")
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(results)} images")
    return results


def format_detections(model_predictions):
    """
    Convert your model's output to the expected detection format.
    
    Expected format for each detection:
    {
        'class': str (e.g., 'Batsman', 'Ball'),
        'confidence': float (0.0-1.0),
        'bbox': [x, y, width, height]
    }
    
    Modify this function to match your model's output structure.
    
    Args:
        model_predictions: Output from your detection model
        
    Returns:
        list: List of detection dictionaries in expected format
    """
    detections = []
    
    # TODO: Implement conversion from your model's output format
    # Example:
    # for pred in model_predictions:
    #     detections.append({
    #         'class': pred['class_name'],
    #         'confidence': float(pred['score']),
    #         'bbox': pred['bbox_coordinates']
    #     })
    
    return detections


def load_model(model_path):
    """
    Load a scikit-learn model from a pickle file with error handling.
    
    Args:
        model_path (str): Path to the pickle file
        
    Returns:
        model: Loaded model object, or None if loading fails
        
    Raises:
        Prints error messages for debugging
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
        
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found: {model_path}")
        print("   Solution: Place your model pickle file in the app directory")
        return None
        
    except ValueError as e:
        if "incompatible dtype" in str(e):
            print("❌ ERROR: scikit-learn version mismatch!")
            print("   The model was created with a different scikit-learn version")
            print("   ")
            print("   Solutions:")
            print("   1. Upgrade scikit-learn: pip install --upgrade scikit-learn")
            print("   2. Or install matching version: pip install scikit-learn==X.Y.Z")
            print("   3. Recreate the pickle file with current scikit-learn version")
            print("   ")
            print(f"   Error details: {str(e)[:200]}...")
        else:
            print(f"❌ ERROR: Value error when loading model: {str(e)}")
        return None
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {type(e).__name__}: {str(e)}")
        return None
