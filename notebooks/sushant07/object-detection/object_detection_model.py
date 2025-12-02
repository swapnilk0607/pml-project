import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

    
categories = [ "ball","bat"]
data = []
labels = []    
data_dir = "dataset"
le = LabelEncoder()
svm = SVC(kernel='rbf', probability=True)
rfc = RandomForestClassifier(random_state=42)

def train():
    for category in categories:
        path = os.path.join(data_dir, category)
        print(path)
        label = category
        for file in os.listdir(path):
            print(file)
            image = imread(path+'\\'+file)
            gray_image = rgb2gray(image)
            resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
            plt.imshow(resized_image)
            plt.show()
            data.append(resized_image)
            labels.append(label)

    print("Total samples:", len(data))

    hog_features = []

    for image in data:
        features = hog(image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys')
        hog_features.append(features)
        
    X = np.array(hog_features)
    y = np.array(labels)


    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )


    print(X_train)
    print(y_train)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
def train_rfc():
    for category in categories:
        path = os.path.join(data_dir, category)
        print(path)
        label = category
        for file in os.listdir(path):
            print(file)
            image = imread(path+'\\'+file)
            gray_image = rgb2gray(image)
            resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
            plt.imshow(resized_image)
            plt.show()
            data.append(resized_image)
            labels.append(label)

    print("Total samples:", len(data))

    hog_features = []

    for image in data:
        features = hog(image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys')
        hog_features.append(features)
        
    X = np.array(hog_features)
    y = np.array(labels)


    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )


    print(X_train)
    print(y_train)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()   
    
def predict_object():
    image = imread("image_12.jpg")
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
    plt.imshow(resized_image)
    plt.show()
    features = hog(resized_image, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    pred = svm.predict([features])[0]
    return le.inverse_transform([pred])[0]

def predict_object_rfc():
    image = imread("image_12.jpg")
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, (800, 600), anti_aliasing=True)
    plt.imshow(resized_image)
    plt.show()
    features = hog(resized_image, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    pred = rfc.predict([features])[0]
    return le.inverse_transform([pred])[0]


if __name__ == "__main__":
    train()
    print(predict_object())
    
    train_rfc()
    print(predict_object_rfc())
    
    