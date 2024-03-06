import tensorflow as tf
import os
import pandas as pd

# Directories
val_dir = "C:\\Users\\Manuel\\Desktop\\ValDB"
model_path = os.path.join("C:\\Users\\Manuel\\Desktop\\Modelo entrenado", "melanoma_classification_model.h5")

# Load the best model
model = tf.keras.models.load_model(model_path)

# Function to get age based on ISIC ID (same as in your training script)
def get_age(isic_id, combined_csv):
    if isic_id in combined_csv['isic_id'].values:
        return combined_csv[combined_csv['isic_id'] == isic_id]['age_approx'].iloc[0]
    return None

# Function to load and preprocess the image (same as in your training script)
def load_and_preprocess_image(file_path, label, age):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return (image, age), label

# Create a dataset from the validation data (similar to your training script)
def create_val_dataset(image_dir, csv_data):
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
    dataset = dataset.map(lambda x, y, z: load_and_preprocess_image(x, y, z),
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(16)

# Load the CSV data
melanoma_csv = pd.read_csv("C:\\Users\\Manuel\\Desktop\\augmented_metadata_melanoma.csv")
benign_csv = pd.read_csv("C:\\Users\\Manuel\\Desktop\\augmented_metadata_benign.csv")
combined_csv = pd.concat([melanoma_csv, benign_csv])

# Create the validation dataset
val_dataset = create_val_dataset(val_dir, combined_csv)

# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")