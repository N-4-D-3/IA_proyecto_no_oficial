import os
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Cargar el modelo previamente entrenado
model = load_model("modelo_simbolos.h5")

# Cargar el mapa de etiquetas
with open("mapa_etiquetas.json", "r") as f:
    indice_a_etiqueta = json.load(f)
etiqueta_a_indice = {v: k for k, v in indice_a_etiqueta.items()}

# Función para normalizar y preprocesar la imagen
def preprocesar_imagen(imagen):
    imagen = imagen.convert('L').resize((28, 28))  # Redimensiona a 28x28 y convierte a escala de grises
    imagen = np.array(imagen) / 255.0  # Normaliza
    imagen = imagen.reshape(1, 28, 28, 1)  # Añade la dimensión extra para que sea compatible con el modelo
    return imagen

# Función para hacer predicciones
def predecir_simbolo(imagen):
    imagen = preprocesar_imagen(imagen)
    prediccion = model.predict(imagen)
    prediccion_idx = np.argmax(prediccion, axis=1)[0]  # Obtiene el índice con la mayor probabilidad
    return indice_a_etiqueta[str(prediccion_idx)]  # Devuelve la etiqueta correspondiente

class SymbolCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recolector de Símbolos")
        self.canvas_width = 400
        self.canvas_height = 200

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.iniciar_dibujo)
        self.canvas.bind("<B1-Motion>", self.dibujar)

        # Variables y widgets para mostrar la predicción
        self.result_var = tk.StringVar()
        tk.Label(root, text="Símbolo Predicho:", font=("Arial", 12)).pack(pady=5)
        self.result_textbox = tk.Text(root, height=2, width=20, font=("Arial", 14), wrap="word")
        self.result_textbox.pack(pady=5)

        # Botón de limpieza
        tk.Button(root, text="Limpiar", command=self.limpiar).pack(pady=10)

        self.folder = "dataset_simbolos"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.dibujando = False
        self.x1, self.y1 = None, None

    def iniciar_dibujo(self, event):
        self.dibujando = True
        self.x1, self.y1 = event.x, event.y

    def dibujar(self, event):
        if self.dibujando:
            x2, y2 = event.x, event.y
            self.canvas.create_line(self.x1, self.y1, x2, y2, width=5, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.x1, self.y1, x2, y2], fill="black", width=5)
            self.x1, self.y1 = x2, y2

            # Comprobamos si se ha dibujado el signo "="
            self.predecir()

    def predecir(self):
        # Predicción sobre la imagen generada
        simbolo_predicho = predecir_simbolo(self.image)
        self.result_textbox.delete(1.0, tk.END)  # Limpiar el TextBox antes de mostrar el nuevo resultado
        self.result_textbox.insert(tk.END, f"Predicción: {simbolo_predicho}")  # Mostrar la predicción en el TextBox

    def limpiar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.result_textbox.delete(1.0, tk.END)  # Limpiar el TextBox

if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolCollectorApp(root)
    root.mainloop()
