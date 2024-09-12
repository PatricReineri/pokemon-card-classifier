import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import json
from card_classifier_usage import PokemonClassifier
import time
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk

def save_screenshot(roi):
    filename = 'screenshot.png'
    cv2.imwrite(filename, roi)
    print(f'Screenshot saved as {filename}')

def get_dominant_color(roi, n_colors=1):
    roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = roi.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening webcam.")
        return False, None

    desired_width = 1280
    desired_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    rect_w, rect_h = 300, 420
    rect_x = desired_width // 4 - rect_w // 2
    rect_y = (desired_height - rect_h) // 2
    
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            cap.release()
            cv2.destroyAllWindows()
            return False, None

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)
        roi = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            print("5 seconds elapsed. Capturing screenshot...")
            save_screenshot(roi)
            cap.release()
            cv2.destroyAllWindows()
            return True, roi

        cv2.imshow('ROI', roi)
        cv2.imshow('Webcam Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            cap.release()
            cv2.destroyAllWindows()
            return False, None

def display_predicted_card(image_path):
    root = tk.Tk()
    root.title("Carta Predetta e Screenshot")
    root.geometry("1200x800")
    root.configure(bg="black")

    frame_images = Frame(root, bg="black")
    frame_images.pack(pady=20)

    frame_pred = Frame(frame_images, bg="black")
    frame_pred.pack(side=tk.LEFT, padx=20)

    frame_screenshot = Frame(frame_images, bg="black")
    frame_screenshot.pack(side=tk.RIGHT, padx=20)

    img_pred = cv2.imread(image_path)
    if img_pred is None:
        print(f"Immagine {image_path} non trovata.")
        root.destroy()
        return

    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    img_pred = Image.fromarray(img_pred)
    img_pred = img_pred.resize((400, 560))
    img_pred_tk = ImageTk.PhotoImage(img_pred)

    label_pred_title = tk.Label(frame_pred, text="Predicted Card", font=("Arial", 18), bg="black", fg="white")
    label_pred_title.pack(pady=10)
    label_pred_img = Label(frame_pred, image=img_pred_tk, bg="black")
    label_pred_img.pack()

    screenshot_img = cv2.imread('screenshot.png')
    if screenshot_img is not None:
        screenshot_img = cv2.cvtColor(screenshot_img, cv2.COLOR_BGR2RGB)
        screenshot_img = Image.fromarray(screenshot_img)
        screenshot_img = screenshot_img.resize((400, 560))
        screenshot_img_tk = ImageTk.PhotoImage(screenshot_img)

        label_screenshot_title = tk.Label(frame_screenshot, text="Screenshot", font=("Arial", 18), bg="black", fg="white")
        label_screenshot_title.pack(pady=10)
        label_screenshot_img = Label(frame_screenshot, image=screenshot_img_tk, bg="black")
        label_screenshot_img.pack()

    frame_button = Frame(root, bg="black")
    frame_button.pack(pady=20)

    def on_next():
        print("Next button pressed")
        root.destroy()

    button_next = tk.Button(frame_button, text="Next", font=("Arial", 16), bg="blue", fg="white", command=on_next)
    button_next.pack()

    root.mainloop()

def classify_and_show_image():
    classifier = PokemonClassifier(model_path="pokemon_classifier_crop_cards.pth", label_encoder_path='classes.npy')
    
    image_path = 'screenshot.png'
    try:
        prediction = classifier.predict_image(image_path)
        print(f'Predicted class: {prediction}')
        
        json_filename = f"{prediction}.json"
        json_path = os.path.join('base1', json_filename)
        
        classifier.print_json_info(json_path)
        
        image_predicted_path = os.path.join('base1_images', f"{prediction}.png")
        display_predicted_card(image_predicted_path)
            
    except Exception as e:
        print(f"An error occurred: {e}")
 
if __name__ == "__main__":
    while True:
        success, roi = capture_image()
        if success and roi is not None:
            classify_and_show_image()

        print("Restarting webcam process...")
