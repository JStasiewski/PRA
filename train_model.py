import tensorflow as tf
from tensorflow.keras import layers

# Paths
train_dir = ".\split_ttv_dataset_type_of_plants\Train_Set_Folder"
val_dir   = ".\split_ttv_dataset_type_of_plants\Validation_Set_Folder"
test_dir  = ".\split_ttv_dataset_type_of_plants\Test_Set_Folder"

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

# 3) Train
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 4) Fine-tune
# base_model.trainable = True
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_ds, validation_data=val_ds, epochs=10)

# 5) Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

# 6) Save
model.save("my_plant_model")
