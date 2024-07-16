import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split



data_dir_train = '/Users/altun/Desktop/dataSet/train'
data_dir_val = '/Users/altun/Desktop/dataSet/val'


def load_data(data_dir, image_size=(512,512)):
    images = []
    masks = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename)) as f:
                polygons = json.load(f)
                mask = create_multi_class_mask_from_polygons(polygons, image_size)

                image_filename = filename.replace('.json', '.jpg')
                image = Image.open(os.path.join(data_dir, image_filename))
                images.append(np.array(image.resize(image_size)))
                masks.append(mask)
    return np.array(images), np.array(masks)


def create_multi_class_mask_from_polygons(polygons, image_size=(512,512)):
    mask = np.zeros(image_size + (3,), dtype=np.uint8)  # Her çizgi için ayrı katman
    try:
        for polygon in polygons:
            if 'label' in polygon and 'points' in polygon:
                label = polygon['label']  # JSON dosyasındaki etiket adını al
                poly_points = polygon['points']
                if label == 'kalp':
                    ImageDraw.Draw(mask[:, :, 0]).polygon(poly_points, outline=1, fill=1)
                elif label == 'akil':
                    ImageDraw.Draw(mask[:, :, 1]).polygon(poly_points, outline=1, fill=1)
                elif label == 'yasam':
                    ImageDraw.Draw(mask[:, :, 2]).polygon(poly_points, outline=1, fill=1)
    except Exception as e:
        print(f"Error processing polygons: {e}")
    return mask


train_images, train_masks = load_data(data_dir_train)
val_images, val_masks = load_data(data_dir_val)

def unet_model(input_size=(512,512, 3), num_classes=3):
    inputs = keras.Input(input_size)
    # Downsampling
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Upsampling
    u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=20, batch_size=8)

import random
import matplotlib.pyplot as plt

# Modeli kaydet
model_path = './unet_model.h5'
model.save(model_path)
print(f"Model {model_path} dosyasına kaydedildi.")

index = random.randint(0, len(val_images) - 1)
test_image = val_images[index]
test_mask_true = val_masks[index]

predicted_mask = model.predict(np.expand_dims(test_image, axis=0))[0]

# Görüntüleri ve maskeleri görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(test_image)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(test_mask_true[:, :, 0], cmap='gray')
plt.title('True Mask')i
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask.argmax(axis=-1), cmap='gray')
plt.title('Predicted Mask')

plt.tight_layout()
plt.show()