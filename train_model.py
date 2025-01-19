import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Paths
train_dir = ".\\split_ttv_dataset_type_of_plants\\Train_Set_Folder"
val_dir   = ".\\split_ttv_dataset_type_of_plants\\Validation_Set_Folder"
test_dir  = ".\\split_ttv_dataset_type_of_plants\\Test_Set_Folder"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# 1) Datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

# 2) Transfer learning model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(30, activation='softmax')  # adjust # of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3) Train and store history
history = model.fit(train_ds, validation_data=val_ds, epochs=20)

# 4) Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

# 5) Save model
model.save("my_plant_model_0.2")

# 6) Plot training history
def plot_history(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f"{filename}_accuracy.png")
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{filename}_loss.png")
    plt.show()

# Save training plots
plot_history(history, "training_plot")

print("Training plots saved as 'training_plot_accuracy.png' and 'training_plot_loss.png'.")
