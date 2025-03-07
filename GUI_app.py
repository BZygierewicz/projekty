import tkinter as tk
from mnist_model import *
from numpy import ndarray
from torchvision.transforms import functional as F
from tkinter import Canvas
from PIL import Image, ImageDraw
import numpy as np

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie cyfr")
        self.digit = None

         # Płótno do rysowania
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Przyciski
        self.btn_predict = tk.Button(root, text="Rozpoznaj", command=self.predict_digit)
        self.btn_predict.pack()
        self.btn_clear = tk.Button(root, text="Wyczyść", command=self.clear_canvas)
        self.btn_clear.pack()

        # Etykieta wyniku
        self.result_label = tk.Label(root, text="Narysuj cyfrę i kliknij 'Rozpoznaj'", font=("Arial", 14))
        self.result_label.pack()

        # Obraz do zapisu rysunku
        self.image = Image.new("L", (280, 280), 255)  # Tło białe
        self.draw = ImageDraw.Draw(self.image)

        # Obsługa rysowania myszką
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def preprocess_image(self,img):
        img = img.convert("L")
        img = img.point(lambda x: 255 - x)  # Odwrócenie kolorów
        img = img.resize((28, 28))
        img = F.to_tensor(img)
        img = F.normalize(img, (0.5,), (0.5,))
        return img.unsqueeze(0)

    def draw_lines(self, event):
        """ Rysowanie grubą linią dla lepszej widoczności cyfry """
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill="black", width=10)
        self.draw.ellipse([x, y, x+10, y+10], fill="black")

    def predict_digit(self):
        """przewidywanie i wyświetlanie cyfry"""
        # Przetwarzanie obrazu
        self.image = self.preprocess_image(self.image)
        # Przekazanie obrazu do modelu
        with torch.no_grad():
            output = model(self.image)
            _, predicted = torch.max(output, 1)

        self.digit = predicted.item()
        print(f"Rozpoznana cyfra: {self.digit}")
        self.result_label.config(text=f"Rozpoznana cyfra: {self.digit}")
        self.digit = None

    def clear_canvas(self):
        """ Czyszczenie płótna i resetowanie obrazu """
        self.canvas.delete("all")  # Czyszczenie płótna
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Narysuj cyfrę i kliknij 'Rozpoznaj'")
