import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
# import pickle
from matplotlib import pyplot as plt
import time

target_size = (150, 150)

def resize_image_with_padding(image, target_size):
    height, width = image.shape[:2]
    target_height, target_width = target_size

    aspect_ratio = float(width) / float(height)
    target_aspect_ratio = float(target_width) / float(target_height)

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        pad_left, pad_right = 0, 0
    else:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top, pad_bottom = 0, 0

    resized_image = cv2.resize(image, (new_width, new_height))
    resized_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return resized_image

def extract_image_features(image_path, labeled_path, gabor_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize_image_with_padding(img, target_size)
    
    features = pd.DataFrame()
    imgOrginal = img.reshape(-1)
    features['Original Image'] = imgOrginal
    
    # LBP
    METHOD = 'uniform'
    radius = 1
    n_points = 6
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    features['LBP'] = lbp.reshape(-1)
            
    #Gabor
    with open(gabor_path, "w") as gabor_results:
        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for lamda in np.arange(0, np.pi, np.pi / 4):
                    for gamma in (0.05, 0.5):
                        gabor_label = 'Gabor' + str(num)
                        ksize=9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        kernels.append(kernel)
                        fimg = cv2.filter2D(imgOrginal, cv2.CV_8UC3, kernel)
                        features[gabor_label] = fimg.reshape(-1)
                        gabor_results.write(gabor_label + ': theta=' + str(theta) + ': sigma=' + str(sigma) + ': lamda=' + str(lamda) + ': gamma=' + str(gamma) + '\n')
                        num += 1
    gabor_results.close()
    
    #CANNY EDGE
    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1)
    features['Canny Edge'] = edges1

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    features['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    features['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    features['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    features['Prewitt'] = edge_prewitt1

    #GAUSSIAN with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    features['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    features['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    features['Median s3'] = median_img1

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    features['Variance s3'] = variance_img1
    
    #Label  
    labeled_img = cv2.imread(labeled_path)
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    labeled_img = resize_image_with_padding(labeled_img, target_size)
    labeled_img1 = labeled_img.reshape(-1)
    features['Labels'] = labeled_img1
    
    return features

def train_feature_extraction():
    df = pd.DataFrame()

    for number_str in range(1, 10):
        image_path = 'images/train_images/subject_' + str(number_str) + '.jpg'
        labeled_path = 'images/train_masks/subject_' + str(number_str) + '.png'
        gaborsPath = 'outputs_MachineLearning/gabors/subject_' + str(number_str) + '.txt'
        features = extract_image_features(image_path, labeled_path, gaborsPath)
        df = pd.concat([df, features], ignore_index = True)

    df.to_csv('outputs_MachineLearning/train_features.csv')
    
    X_train = df.drop(labels = ["Labels"], axis=1)
    labeled_df = pd.read_csv('outputs_MachineLearning/train_features.csv')
    Y_train = labeled_df['Labels'].values

    return X_train, Y_train

def test_feature_extraction():
    image_path = 'images/test_image/subject_10.jpg'
    labeled_path = 'images/test_mask/subject_10.png'
    gaborsPath = 'outputs_MachineLearning/gabors/subject_10.txt'
    
    features = extract_image_features(image_path, labeled_path, gaborsPath)
    features.to_csv('outputs_MachineLearning/test_features.csv')

    X_test = features.drop(labels = ["Labels"], axis=1)
    labeled_df = pd.read_csv('outputs_MachineLearning/test_features.csv')
    Y_test = labeled_df['Labels'].values

    return X_test, Y_test

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

X_train, Y_train = train_feature_extraction()
X_test, Y_test = test_feature_extraction()

print('\nFEATURE EXTRACTION COMPLETED...')
print('\nDATA SPLITTED...')

X_train_subset = X_train
X_test_subset = X_test

# RandomForest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_start = time.time()
random_forest_model.fit(X_train_subset, Y_train)
rf_end = time.time()
rf_time = rf_end - rf_start

random_forest_predict_train = random_forest_model.predict(X_train_subset)
random_forest_accuracy_train = metrics.accuracy_score(Y_train, random_forest_predict_train)
random_forest_predict_test = random_forest_model.predict(X_test_subset)
random_forest_accuracy_test = metrics.accuracy_score(Y_test, random_forest_predict_test)
random_forest_IoU_test = calculate_iou(Y_test, random_forest_predict_test)

with open('outputs_MachineLearning/random_forest.txt', "w") as random_forest_results:
    random_forest_results.write(f"RandomForest Train Accuracy: {random_forest_accuracy_train:.4f}\n")
    random_forest_results.write(f"RandomForest Test Accuracy: {random_forest_accuracy_test:.4f}\n")
    random_forest_results.write(f"RandomForest Train Time: {rf_time:.4f} seconds\n")
    random_forest_results.write(f"RandomForest IoU on test data: {random_forest_IoU_test:.4f}")
random_forest_results.close()
print('\nRandomForest COMPLETED...')

# AdaBoost
ada_boost_model = AdaBoostClassifier(n_estimators=100, random_state=42)

ab_start = time.time()
ada_boost_model.fit(X_train_subset, Y_train)
ab_end = time.time()
ab_time = ab_end - ab_start

ada_boost_predict_train = ada_boost_model.predict(X_train_subset)
ada_boost_accuracy_train = metrics.accuracy_score(Y_train, ada_boost_predict_train)
ada_boost_predict_test = ada_boost_model.predict(X_test_subset)
ada_boost_accuracy_test = metrics.accuracy_score(Y_test, ada_boost_predict_test)
ada_boost_IoU_test = calculate_iou(Y_test, ada_boost_predict_test)

with open('outputs_MachineLearning/ada_boost.txt', "w") as ada_boost_results:
    ada_boost_results.write(f"AdaBoost Train Accuracy: {ada_boost_accuracy_train:.4f}\n")
    ada_boost_results.write(f"AdaBoost Test Accuracy: {ada_boost_accuracy_test:.4f}\n")
    ada_boost_results.write(f"AdaBoost Train Time: {ab_time:.4f} seconds\n")
    ada_boost_results.write(f"AdaBoost IoU on test data: {ada_boost_IoU_test:.4f}")
ada_boost_results.close()
print('\nAdaBoost COMPLETED...')

# LightGBM
lightgbm_model = LGBMClassifier(n_estimators=100, random_state=42)

lgbm_start = time.time()
lightgbm_model.fit(X_train_subset, Y_train)
lgbm_end = time.time()
lgbm_time = lgbm_end - lgbm_start

lightgbm_predict_train = lightgbm_model.predict(X_train_subset)
lightgbm_accuracy_train = metrics.accuracy_score(Y_train, lightgbm_predict_train)
lightgbm_predict_test = lightgbm_model.predict(X_test_subset)
lightgbm_accuracy_test = metrics.accuracy_score(Y_test, lightgbm_predict_test)
lightgbm_IoU_test = calculate_iou(Y_test, lightgbm_predict_test)

with open('outputs_MachineLearning/lightgbm.txt', "w") as lightgbm_results:
    lightgbm_results.write(f"LightGBM Train Accuracy: {lightgbm_accuracy_train:.4f}\n")
    lightgbm_results.write(f"LightGBM Test Accuracy: {lightgbm_accuracy_test:.4f}\n")
    lightgbm_results.write(f"LightGBM Train Time: {lgbm_time:.4f} seconds\n")
    lightgbm_results.write(f"LightGBM IoU on test data: {lightgbm_IoU_test:.4f}")
lightgbm_results.close()
print('\nLightGBM COMPLETED...')

# CatBoost
cat_boost_model = CatBoostClassifier(n_estimators=100, random_state=42, verbose=False)

cat_boost_start = time.time()
cat_boost_model.fit(X_train_subset, Y_train)
cat_boost_end = time.time()
cat_boost_time = cat_boost_end - cat_boost_start

cat_boost_predict_train = cat_boost_model.predict(X_train_subset)
cat_boost_accuracy_train = metrics.accuracy_score(Y_train, cat_boost_predict_train)
cat_boost_predict_test = cat_boost_model.predict(X_test_subset)
cat_boost_accuracy_test = metrics.accuracy_score(Y_test, cat_boost_predict_test)
cat_boost_IoU_test = calculate_iou(Y_test, cat_boost_predict_test)

with open('outputs_MachineLearning/cat_boost.txt', "w") as cat_boost_results:
    cat_boost_results.write(f"CatBoost Train Accuracy: {cat_boost_accuracy_train:.4f}\n")
    cat_boost_results.write(f"CatBoost Test Accuracy: {cat_boost_accuracy_test:.4f}\n")
    cat_boost_results.write(f"CatBoost Train Time: {cat_boost_time:.4f} seconds\n")
    cat_boost_results.write(f"CatBoost IoU on test data: {cat_boost_IoU_test:.4f}")
cat_boost_results.close()
print('\nCatBoost COMPLETED...')

models = {
    'RandomForest': random_forest_accuracy_test,
    'AdaBoost': ada_boost_accuracy_test,
    'LightGBM': lightgbm_accuracy_test,
    'CatBoost': cat_boost_accuracy_test
}

best_model = max(models, key=models.get)
best_accuracy = models[best_model]

# best_model_path = 'outputs_MachineLearning/best_model.pkl'
if best_model == 'RandomForest':
    # pickle.dump(random_forest_model, open(best_model_path, 'wb'))
    prediction = random_forest_predict_test
    best_model_name = 'RandomForest'
elif best_model == 'AdaBoost':
    # pickle.dump(ada_boost_model, open(best_model_path, 'wb'))
    prediction = ada_boost_predict_test
    best_model_name = 'AdaBoost'
elif best_model == 'LightGBM':
    # pickle.dump(lightgbm_model, open(best_model_path, 'wb'))
    prediction = lightgbm_predict_test
    best_model_name = 'LightGBM'
elif best_model == 'CatBoost':
    # pickle.dump(cat_boost_model, open(best_model_path, 'wb'))
    prediction = cat_boost_predict_test
    best_model_name = 'CatBoost'
    
with open('outputs_MachineLearning/best_model.txt',"w") as best_model:
    best_model.write('Best Model Name: ' + best_model_name + '\n')
    best_model.write(f"Accuracy: {best_accuracy:.4f}")
best_model.close()
    
print('\nBEST MODEL FOUND...')

img = cv2.imread('images/test_image/subject_10.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = resize_image_with_padding(img, target_size)
labeled_img = cv2.imread('images/test_mask/subject_10.png')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img = resize_image_with_padding(labeled_img, target_size)
test_segmented_path = 'images/segmented_image_MachineLearning/subject_10.png'

segmented = prediction.reshape(img.shape)

plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(labeled_img, cmap='jet')
plt.subplot(224)
plt.imshow(segmented, cmap='jet')
plt.imsave(test_segmented_path, segmented, cmap='jet')

print('\nSEGMENTATION COMPLETED')
