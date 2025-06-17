import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import os

# Create the main window
window = tk.Tk()
window.title("License Plates Database")

# Set the window size to full screen
window.geometry("800x600")

# Load the background image
background_image_path = "/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles/18015-MC20BluInfinito-scaled-e1707920217641.jpg"
image = Image.open(background_image_path)
image = image.resize((800, 600), Image.LANCZOS)
background_image = ImageTk.PhotoImage(image)

# Create a label to hold the background image
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Function to execute the speed_detection.py script
def speed_detection():
    subprocess.run(["python", "speed_detection.py"])

# Function to execute the helmet_detection.py script
def helmet_detection():
    subprocess.run(["python", "helmet_detection.py"])

# Function to open the detected_frames folder (Linux compatible)
def open_violations_folder():
    folder_path = "/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles/license_plates"
    if os.path.exists(folder_path):
        subprocess.run(["xdg-open", folder_path])
    else:
        print(f"Folder '{folder_path}' does not exist.")

# Function to execute the speed_parameters.py script
def speed_parameters():
    subprocess.run(["python", "speed_parameters.py"])

# Helper function to change the button color on hover
def on_enter(e):
    e.widget['background'] = '#ADD8E6'  # Light blue on hover

def on_leave(e):
    e.widget['background'] = '#D3D3D3'  # Light gray when not hovered

# Create stylish buttons and apply hover effects
def create_button(text, command):
    button = tk.Button(window, text=text, width=30, height=2, command=command,
                       bg='#D3D3D3',  # Light gray background
                       fg='black',  # Black text
                       bd=5,  # Border width
                       relief='raised',  # Raised button style
                       font=('Helvetica', 12, 'bold'))  # Font style
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    return button

# Center the buttons on the screen using grid layout
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(5, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create buttons
button1 = create_button("Speed Detection", speed_detection)
button1.grid(row=1, column=0, pady=10)

button_parameters = create_button("Speed Parameters", speed_parameters)
button_parameters.grid(row=2, column=0, pady=10)

button2 = create_button("Helmet Detection", helmet_detection)
button2.grid(row=3, column=0, pady=10)

button3 = create_button("Violations", open_violations_folder)
button3.grid(row=4, column=0, pady=10)

# Run the application
window.mainloop()
