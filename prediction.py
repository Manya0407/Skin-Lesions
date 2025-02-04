import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("skin_lesion_model_continued.h5")

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Path to the image you want to predict
img_path = r"C:\Users\saura\Desktop\skin lesions\dataset\test\Actinic keratoses\ISIC_0030036.jpg" # Change to your image path

# Preprocess the image
input_image = preprocess_image(img_path)

# Make a prediction
prediction = model.predict(input_image)

# Interpret the result
threshold = 0.5  # Assuming a binary classification (Malignant vs Benign)
if prediction[0][0] > threshold:
    print("Prediction: Malignant (Positive for skin lesion)")
else:
    print("Prediction: Benign (Negative for skin lesion)")

# Print raw probability score
print(f"Probability Score: {prediction[0][0]:.4f}")
