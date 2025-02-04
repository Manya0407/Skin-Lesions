from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the training data generator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'path_to_train_directory',  # specify the path to your training data directory
    target_size=(64, 64),       # resize images as needed
    batch_size=32,
    class_mode='binary'         # or 'categorical' if it's a multi-class problem
)

# Define the validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'val',  # specify the path to your validation data directory
    target_size=(64, 64),     # resize images as needed
    batch_size=32,
    class_mode='binary'       # or 'categorical' if it's a multi-class problem
)

# Load the previously saved model
from tensorflow.keras.models import load_model
model = load_model("skin_lesion_model.h5")

# Compile the model again if needed
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training for more epochs
additional_epochs = 5
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=additional_epochs,
    initial_epoch=history.epoch[-1]  # Continue from the last epoch
)

# Optionally save the model again
model.save("skin_lesion_model_continued.h5")

print("Training continued. Model saved as skin_lesion_model_continued.h5")
