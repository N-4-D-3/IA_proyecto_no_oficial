import tkinter as tk
from PIL import Image, ImageDraw
import os

# Función para convertir símbolos en nombres válidos
def normalizar_etiqueta(etiqueta):
    reemplazos = {
        '*': 'mult',
        '/': 'div',
        '+': 'plus',
        '-': 'minus',
        '=': 'equal',
        '÷': 'div2',
        'x': 'mult_x',
    }
    
    # Si la etiqueta es un número, devolver 'num_'
    if etiqueta.isdigit():
        return 'num_' + etiqueta  # Los números se etiquetan con 'num_'
    
    return reemplazos.get(etiqueta, etiqueta)

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

        self.canvas.bind("<B1-Motion>", self.paint)

        self.label_var = tk.StringVar()
        tk.Entry(root, textvariable=self.label_var, font=("Arial", 12)).pack(pady=10)
        tk.Button(root, text="Guardar", command=self.guardar).pack(pady=10)
        tk.Button(root, text="Limpiar", command=self.limpiar).pack()

        self.folder = "dataset_simbolos"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def guardar(self):
        etiqueta_original = self.label_var.get()
        if not etiqueta_original:
            print("¡Escribe una etiqueta primero!")
            return

        etiqueta = normalizar_etiqueta(etiqueta_original)
        existing = len([f for f in os.listdir(self.folder) if f.startswith(etiqueta)])
        path = os.path.join(self.folder, f"{etiqueta}_{existing}.png")
        self.image.save(path)
        print(f"Guardado: {path}")
        self.limpiar()

    def limpiar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")

if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolCollectorApp(root)
    root.mainloop()