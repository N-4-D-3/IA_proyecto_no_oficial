import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carpeta del dataset
DATASET_DIR = "dataset_simbolos"

# Etiquetas válidas (normalizadas)
etiquetas_validas = sorted(set(f.split('_')[0] for f in os.listdir(DATASET_DIR)))
etiqueta_a_indice = {etiqueta: idx for idx, etiqueta in enumerate(etiquetas_validas)}
indice_a_etiqueta = {v: k for k, v in etiqueta_a_indice.items()}

# Guardar mapa para usar luego en el main
import json
with open("mapa_etiquetas.json", "w") as f:
    json.dump(indice_a_etiqueta, f)

# Leer imágenes y etiquetas
X = []
y = []

for archivo in os.listdir(DATASET_DIR):
    etiqueta = archivo.split('_')[0]
    if etiqueta in etiqueta_a_indice:
        path = os.path.join(DATASET_DIR, archivo)
        imagen = Image.open(path).convert('L').resize((28, 28))
        imagen = np.array(imagen)
        imagen = imagen / 255.0  # Normalización más simple
        X.append(imagen)
        y.append(etiqueta_a_indice[etiqueta])

X = np.array(X).reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=len(etiqueta_a_indice))

# Separar entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un generador de aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Modelo CNN mejorado
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(etiqueta_a_indice), activation='softmax')
])

# Optimizar con Adam y usar un Learning Rate Scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ajustar el learning rate cuando no haya mejoras
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Entrenar el modelo con aumento de datos
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Guardar modelo
model.save("modelo_simbolos.h5")

print("✅ ¡Modelo entrenado y guardado como 'modelo_simbolos.h5'!")
