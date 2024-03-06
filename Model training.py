import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import numpy as np
import pandas as pd
import os
import json

# Set logging verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Check for GPU availability
if tf.config.experimental.list_physical_devices('GPU'):
    print('Using GPU')
else:
    print('Using CPU')


# Function to get age based on ISIC ID
def get_age(isic_id, combined_csv):
    if isic_id in combined_csv['isic_id'].values:
        return combined_csv[combined_csv['isic_id'] == isic_id]['age_approx'].iloc[0]
    return None


# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])



# Function to load and preprocess the image
def load_and_preprocess_image(file_path, label, age, augment=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])

    if augment:
        image = data_augmentation(image)

    return (image, age), label

# Create a dataset of file paths and labels
def create_dataset(image_dir, csv_data, augment=False):
    image_paths, labels, ages = [], [], []
    for classify_type in ['melanoma', 'benign']:
        class_dir = os.path.join(image_dir, classify_type)
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            isic_id = os.path.splitext(filename)[0]
            age = get_age(isic_id, csv_data)
            if age is not None:
                image_paths.append(file_path)
                labels.append(1 if classify_type == 'melanoma' else 0)
                ages.append(float(age))

    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)
    ages = tf.constant(ages)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, ages))
    dataset = dataset.map(lambda x, y, z: load_and_preprocess_image(x, y, z, augment),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


# Directories and CSV loading
train_dir = "C:\\Users\\Manuel\\Desktop\\TrainDB"
val_dir = "C:\\Users\\Manuel\\Desktop\\ValDB"
melanoma_csv = pd.read_csv("C:\\Users\\Manuel\\Desktop\\augmented_metadata_melanoma.csv")
benign_csv = pd.read_csv("C:\\Users\\Manuel\\Desktop\\augmented_metadata_benign.csv")
model_dir = "C:\\Users\\Manuel\\Desktop\\Modelo entrenado"
model_path = os.path.join(model_dir, "melanoma_classification_model.h5")
history_path = os.path.join(model_dir, "history.json")

combined_csv = pd.concat([melanoma_csv, benign_csv])

# Creating the training and validation datasets
train_dataset = create_dataset(train_dir, combined_csv, augment=True)

val_dataset = create_dataset(val_dir, combined_csv, augment=False)

# Shuffling, batching, and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(10000).batch(16).prefetch(AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(AUTOTUNE)


# Load pre-trained ResNet152 model
base_model = tf.keras.applications.ResNet152(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True
fine_tune_at = 130
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# L1 and L2 regularizers
l1_l2_reg = regularizers.l1_l2(l1=1e-3, l2=1e-2)

# New input for age
input_age = tf.keras.Input(shape=(1,), name='age_input')
age_dense = layers.Dense(10, activation='relu')(input_age)

# Concatenate age input with CNN output
base_model_output = layers.GlobalAveragePooling2D()(base_model.output)
concatenated = layers.concatenate([base_model_output, age_dense])

# Output layer
output = layers.Dense(1, activation='sigmoid', kernel_regularizer=l1_l2_reg)(concatenated)

# Final model
model = models.Model(inputs=[base_model.input, input_age], outputs=output)

# Compile with Nadam optimizer
optimizer = tf.keras.optimizers.Nadam(learning_rate=5e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Custom Model Checkpoint
class CustomModelCheckpoint(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(model_path)
        history_dict = {k: [float(v) for v in values] for k, values in self.model.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
        print(f"Checkpoint saved at {model_path} and history at {history_path}")



# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


# Calculate the number of steps per epoch for training and validation
train_steps_per_epoch = np.ceil((len(os.listdir(os.path.join(train_dir, 'melanoma'))) +
                                len(os.listdir(os.path.join(train_dir, 'benign')))) / 16)
val_steps_per_epoch = np.ceil((len(os.listdir(os.path.join(val_dir, 'melanoma'))) +
                              len(os.listdir(os.path.join(val_dir, 'benign')))) / 16)

# Custom Model Checkpoint to save the best model based on validation accuracy
best_model_checkpoint = callbacks.ModelCheckpoint(
    model_path,
    save_best_only=True,
    monitor='val_accuracy',  # Monitoring validation accuracy
    mode='max'  # Save the model with max validation accuracy
)

# Train the model
model.fit(
    train_dataset,
    epochs=80,
    validation_data=val_dataset,
    callbacks=[early_stopping, best_model_checkpoint, reduce_lr]
)

# Manually load the best model
model.load_weights(model_path)


print("Model training completed and saved!")