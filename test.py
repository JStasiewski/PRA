import tensorflow as tf
import numpy as np
from PIL import Image

# 1) Load your SavedModel (assuming you used model.save("my_plant_model"))
model = tf.keras.models.load_model("my_plant_model")

# 2) Helper function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    # Open image
    img = Image.open(image_path).convert("RGB")  
    # Resize to match model input
    img = img.resize(target_size)
    # Convert to NumPy array
    img_array = np.array(img)
    # Scale/normalize pixels if needed
    img_array = img_array / 255.0
    # Add a batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

# 3) Predict on a sample image
sample_image_path = "image.png"  # change to an actual image path
x = preprocess_image(sample_image_path)
predictions = model.predict(x)

# 4) Interpret the predictions
# Suppose you have 4 classes: [class0, class1, class2, class3]
class_names = ["aloevera", "banan", "bilimbi", "cantaloupe", "cassava", "coconut", "corn", "cucumber", 
    "curcuma", "eggplant", "galangal", "ginger", "guava", "kale", 
    "longbeans", "mango", "melon", "orange", "paddy", "papaya", 
    "peper chili", "pineapple", "pomelo", "shallot", "soybeans", 
    "spinach", "sweet potatoes", "tobacco", "waterapple", "watermelon"
]

# predictions is shape (1, 4). 
# Argmax to get the top class index
predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index]
predicted_class = class_names[predicted_index]

print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
