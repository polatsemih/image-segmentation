import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
import time

def unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_data(train_images_dir, train_masks_dir, test_image_dir, test_mask_dir):
    train_images = []
    train_masks = []

    for i in range(1, 10):
        image_path = os.path.join(train_images_dir, f"subject_{i}.jpg")
        mask_path = os.path.join(train_masks_dir, f"subject_{i}.png")
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        # convert mask to single-channel representation
        mask = np.expand_dims(mask[:, :, 0], axis=-1)

        train_images.append(image)
        train_masks.append(mask)

    test_image_path = os.path.join(test_image_dir, "subject_10.jpg")
    test_mask_path = os.path.join(test_mask_dir, "subject_10.png")
    test_image = np.array(Image.open(test_image_path))
    test_mask = np.array(Image.open(test_mask_path))

    # convert mask to single-channel representation
    test_mask = np.expand_dims(test_mask[:, :, 0], axis=-1)

    return train_images, train_masks, test_image, test_mask

def semantic_segmentation(train_images, train_masks, test_image, test_mask):
    # Preprocess images and masks
    train_images = np.array(train_images) / 255.0
    train_masks = np.array(train_masks) / 255.0
    test_image = np.array(test_image) / 255.0
    test_mask = np.array(test_mask) / 255.0

    # Define model and compile it
    model = unet((None, None, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    # Train the model and measure training time
    start_time = time.time()
    history = model.fit(train_images, train_masks, epochs=10, batch_size=1)
    end_time = time.time()
    train_time = end_time - start_time

    # Predict on test image
    segmented_image = model.predict(np.expand_dims(test_image, axis=0))[0]

    # Convert predicted output and true mask to single-channel representation
    segmented_image = np.squeeze(segmented_image)
    test_mask = np.squeeze(test_mask)

    # Calculate training accuracy
    train_accuracy = history.history['accuracy'][-1]

    # Evaluate model on test dataset and calculate test accuracy
    test_loss, test_accuracy = model.evaluate(np.expand_dims(test_image, axis=0), np.expand_dims(test_mask, axis=0))

    return train_accuracy, test_accuracy, train_time, segmented_image


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

train_images_dir = "images/train_images/"
train_masks_dir = "images/train_masks/"
test_image_dir = "images/test_image/"
test_mask_dir = "images/test_mask/"

train_images, train_masks, test_image, test_mask = load_data(train_images_dir, train_masks_dir, test_image_dir, test_mask_dir)

train_accuracy, test_accuracy, train_time, segmented_image = semantic_segmentation(train_images, train_masks, test_image, test_mask)

unet_IoU_test = calculate_iou(test_mask, segmented_image)

with open('outputs_DeepLearning/u_net.txt', "w") as unet_results:
    unet_results.write(f"U-Net Training Accuracy: {train_accuracy:.4f}\n")
    unet_results.write(f"U-Net Test Accuracy: {test_accuracy:.4f}\n")
    unet_results.write(f"U-Net Training Time: {train_time:.4f} seconds\n")
    unet_results.write(f"U-Net IoU on Test Data: {unet_IoU_test:.4f}")
unet_results.close()

print('\nU-Net COMPLETED...')

segmented_image = Image.fromarray((segmented_image * 255).astype(np.uint8))
segmented_image.save('images/segmented_image_DeepLearing/subject_10.jpg')

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(test_image)
plt.title("Original Test Image")
plt.subplot(132)
plt.imshow(test_mask, cmap='gray')
plt.title("Mask Test Image")
plt.subplot(133)
plt.imshow(segmented_image, cmap='gray')
plt.title("Segmented Test Image")
plt.show()

print('\nSEGMENTATION COMPLETED')
