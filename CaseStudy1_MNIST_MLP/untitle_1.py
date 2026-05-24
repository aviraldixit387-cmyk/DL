import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from skimage.feature import hog, local_binary_pattern

# 1. SETTINGS

IMG_SIZE = 128
ZIP_FILE = "dataset.zip"     # Optional Kaggle ZIP
DATASET_FOLDER = "/kaggle/input/datasets/omkargurav/face-mask-dataset/data"  # Target extraction path

# 2. FEATURE EXTRACTION

def extract_features(img_uint8):
    """Extracts a massive combined feature vector from a single uint8 image."""
    
    # 1. HSV Color Histogram
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    color_hist = cv2.calcHist([hue], [0], None, [32], [0, 256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # Base Grayscale for Texture and Shape
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # 2. HOG (Histogram of Oriented Gradients)
    hog_feature = hog(
        gray_eq, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), feature_vector=True
    )

    # 3. LBP (Local Binary Pattern)
    lbp = local_binary_pattern(gray_eq, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # 4. Edge Feature (Amount of sharp edges)
    edges = cv2.Canny(gray_eq, 100, 200)
    edge_feature = np.array([np.sum(edges) / 255.0])

    # 5. Shape Features (Fixed: Using Threshold instead of Canny)
    _, thresh = cv2.threshold(gray_eq, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area, perimeter, circularity = 0, 0, 0
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter != 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

    shape_features = np.array([area, perimeter, circularity])

    # Combine all features
    return np.hstack([hog_feature, lbp_hist, color_hist, edge_feature, shape_features])

# 3. DATASET LOADING & AUGMENTATION

def load_and_preprocess(dataset_path):
    """Loads images, applies augmentation, and extracts features."""
    data, labels = [], []
    classes = sorted(os.listdir(dataset_path))
    print(f"Detected Classes: {classes}")

    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing '{class_name}'...")
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            # Load and handle corruption
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Base Preprocessing
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.medianBlur(img, 3)
            # Notice we skip dividing by 255.0 to keep it as uint8 for OpenCV!

            # 1. Extract Original Image Features
            data.append(extract_features(img))
            labels.append(label_idx)

            # 2. Augmentation: Horizontal Flip
            img_flipped = cv2.flip(img, 1)
            data.append(extract_features(img_flipped))
            labels.append(label_idx)

    return np.array(data), np.array(labels), classes

# 4. MODEL SELECTION

def find_best_model(X_train, y_train, X_test, y_test):
    """Compares multiple traditional ML models to find the top performer."""
    models = {
        "SVM (RBF)": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model, best_name, best_acc = None, "", 0
    
    print("\n--- Model Competition ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name
            
    print(f"\n>> WINNER: {best_name} with {best_acc * 100:.2f}% accuracy <<")
    return best_model, best_name

# 5. VISUALIZATION

def plot_learning_curve(estimator, title, X, y, cv=5):
    """Plots the learning curve to check for underfitting/overfitting."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Score")
    plt.legend(loc="best")
    plt.show()

# MAIN EXECUTION PIPELINE

if _name_ == "_main_":
    
    # 1. Handle ZIP Extraction (Kaggle Ready)
    if os.path.exists(ZIP_FILE) and not os.path.exists(DATASET_FOLDER):
        print("Extracting ZIP...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATASET_FOLDER)
        print("ZIP Extracted Successfully.")
    else:
        print("Using Existing Dataset Folder.")

    dataset_path = DATASET_FOLDER

    # 2. Extract Features
    X, y, classes = load_and_preprocess(dataset_path)
    print(f"\nFinal Dataset Shape (with Augmentation): {X.shape}")

    # 3. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling (Crucial for SVM/KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X) # Needed for learning curve

    # 5. Train and Select Best Model
    best_model, best_model_name = find_best_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # 6. Final Evaluation
    y_pred = best_model.predict(X_test_scaled)
    
    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    # 7. Visualizations
    print("Plotting Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, ax=ax, cmap='Blues'
    )
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.show()

    print("Plotting Learning Curve...")
    plot_learning_curve(best_model, f"Learning Curve: {best_model_name}", X_scaled, y)
