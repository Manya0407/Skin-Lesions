import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the correct dataset paths
dataset_dir = "C:/Users/saura/Desktop/skin lesions/dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

# Check if the directories exist before proceeding
if not os.path.exists(train_dir):
    print(f"Error: {train_dir} not found.")
    exit()

if not os.path.exists(val_dir):
    print(f"Error: {val_dir} not found.")
    exit()

# Image Data Generators for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Load the previously saved model
model = load_model("skin_lesion_model.h5")

# Compile the model again if needed (same optimizer, loss, and metrics as before)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training for 5 more epochs
additional_epochs = 5
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=additional_epochs
)

# Save the model after continuing training
model.save("skin_lesion_model_continued.h5")

print("Training continued for 5 more epochs. Model saved as skin_lesion_model_continued.h5")
