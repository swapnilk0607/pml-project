from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ... (load and extract HOG features for positive and negative samples)

    # Load and preprocess image
    
def generate_hog_fetures():
    image = imread('test.jpg')
    gray_image = rgb2gray(image)
    resized_image = resize(gray_image, (800, 600), anti_aliasing=True) # Example size
    # Extract HOG features
    hog_features, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), transform_sqrt=True,
                                  visualize=True, feature_vector=True)
    return hog_features, hog_image




# Assuming 'X' contains HOG features and 'y' contains corresponding labels (0 or 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Linear SVM classifier
# classifier = LinearSVC(random_state=42)
# classifier.fit(X_train, y_train)

# # Evaluate the classifier
# accuracy = classifier.score(X_test, y_test)
# print(f"Classifier accuracy: {accuracy}")


if __name__ == "__main__":
    hog_features, hog_image = generate_hog_fetures()
    print(hog_features.shape)
    plt.imshow(hog_image)
    plt.show()