import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load your trained model
model_path = 'C:\\Users\\Manuel\\Desktop\\Modelo entrenado\\melanoma_classification_model.h5'
model = tf.keras.models.load_model(model_path)
loaded_image = None

def load_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if not file_path:
        messagebox.showinfo("Info", "No image selected.")
        return

    try:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_show = ImageTk.PhotoImage(img)
        panel.config(image=img_show)
        panel.image = img_show
        img = np.array(img)
        if img.shape == (224, 224, 3):
            loaded_image = np.expand_dims(img, axis=0)
        else:
            messagebox.showerror("Error", "The image is not in the correct format.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def predict():
    global loaded_image
    if loaded_image is None:
        messagebox.showerror("Error", "Please load an image first.")
        return

    try:
        age = int(age_entry.get())
        if age < 0 or age > 120:
            messagebox.showerror("Error", "Please enter a valid age (0-120).")
            return
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for age.")
        return

    prediction = model.predict([loaded_image, np.array([[age]])])
    result = "Melanoma" if prediction[0][0] > 0.5 else "Benign"
    result_label.config(text=f"Prediction: {result}")

def close_app():
    root.destroy()

# Set up the main application window
root = tk.Tk()
root.title("Melanoma Detection")
root.geometry("350x500")
root.configure(bg='white')

# Styling the buttons
style = ttk.Style(root)
style.configure('TButton', font=('Helvetica', 12, 'bold'), borderwidth='0', relief='flat', background='light gray')
style.map('TButton', foreground=[('!active', 'black'), ('active', 'blue')], background=[('!active', 'light gray'), ('active', 'white')])

# Create and style widgets
open_button = ttk.Button(root, text="Open Image", command=load_image, style='TButton')
open_button.pack(pady=10)

panel = tk.Label(root, bg='white')
panel.pack(pady=10)

age_label = tk.Label(root, text="Enter Age:", bg='white', font=('Helvetica', 10, 'bold'))
age_label.pack()

age_entry = tk.Entry(root, font=('Helvetica', 10))
age_entry.pack(pady=5)

predict_button = ttk.Button(root, text="Predict", command=predict, style='TButton')
predict_button.pack(pady=10)

result_label = tk.Label(root, text="", bg='white', font=('Helvetica', 10, 'bold'))
result_label.pack(pady=10)

exit_button = ttk.Button(root, text="Exit", command=close_app, style='TButton')
exit_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
