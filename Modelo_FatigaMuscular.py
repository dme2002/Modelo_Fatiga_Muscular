import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import shap
import plotly.express as px  # Para gráficos interactivos
import time

# Cargar datos
def cargar_datos_emg(archivo):
    datos = pd.read_csv(archivo, sep=r"\s+", header=None)
    return datos

# Preprocesamiento avanzado
def preprocesar_datos(datos):
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(datos)
    X = datos_normalizados[:, :-1]  # Características
    y = (datos_normalizados[:, -1] > 0.5).astype(int)  # Etiquetas binarizadas
    return X, y

# Crear modelo de red neuronal avanzada
def crear_modelo_optimizado(input_shape):
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

# Graficar métricas de entrenamiento
def graficar_historial(historial):
    plt.plot(historial.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()
    
    plt.plot(historial.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Precisión de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

# Entrenamiento y evaluación con validación cruzada
def entrenar_y_evaluar_modelo(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    puntajes_accuracy, puntajes_auc, puntajes_f1 = [], [], []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        modelo = crear_modelo_optimizado((X_train.shape[1],))

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        # Tiempo de cada época
        start_time = time.time()
        historial = modelo.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        end_time = time.time()
        print(f"Tiempo por epoch: {(end_time - start_time) / len(historial.history['loss']):.2f} segundos")

        # Evaluación del modelo
        y_pred_prob = modelo.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convertir a binario para cálculo de métricas

        puntajes_accuracy.append(accuracy_score(y_val, y_pred))
        puntajes_auc.append(roc_auc_score(y_val, y_pred_prob))
        puntajes_f1.append(f1_score(y_val, y_pred))

        # Gráficos de métricas por iteración
        graficar_historial(historial)

    print("Puntajes de Validación Cruzada")
    print("Precisión Promedio:", np.mean(puntajes_accuracy))
    print("AUC Promedio:", np.mean(puntajes_auc))
    print("F1-Score Promedio:", np.mean(puntajes_f1))

    return modelo

# Visualización de importancia de características
def visualizacion_importancia_caracteristicas(modelo, X_sample, feature_names):
    # Obtener las importancias de las características con SHAP
    explainer = shap.KernelExplainer(modelo.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Calcular la media absoluta de los valores SHAP
    shap_mean_values = np.mean(np.abs(shap_values), axis=0)

    # Aplanar los valores SHAP para asegurarse de que sean 1D
    shap_mean_values = np.ravel(shap_mean_values)

    # Crear un DataFrame para ordenar las importancias
    importance_df = pd.DataFrame({
        'Característica': feature_names,
        'Porcentaje': shap_mean_values
    }).sort_values(by='Porcentaje', ascending=False)

    # Graficar la importancia de las características con plotly
    fig = px.bar(
        importance_df, 
        x='Porcentaje', 
        y='Característica', 
        orientation='h',
        title="Datos finales sobre el análisis de EMG",
        labels={"Porcentaje": "", "Característica": "Característica"}
    )
    fig.update_traces(hovertemplate='<b>Característica</b>: %{y}<br><b>Porcentaje</b>: %{x:.2f}')
    fig.show()

# Conclusión basada en las predicciones
def conclusion_fatiga(modelo, X_test, y_test):
    # Predicción
    y_pred_prob = modelo.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calcular precisión y tomar una decisión
    precision = accuracy_score(y_test, y_pred)
    if precision >= 0.75:
        fatiga_indicada = "sí"
    else:
        fatiga_indicada = "no"

    # Mostrar la conclusión al usuario
    print(f"\nConclusión sobre la fatiga muscular:")
    print(f"La precisión del modelo es: {precision:.2f}")
    print(f"Según el modelo, ¿tiene fatiga muscular? {fatiga_indicada.capitalize()}")

# Función principal
def main():
    archivo_emg = 'D:/TAREAS UNAH 2024/Inteligencia Artificial/Dataset/sub3.txt'  # Cambia la ruta al archivo correcto
    datos_emg = cargar_datos_emg(archivo_emg)
    
    X, y = preprocesar_datos(datos_emg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generar nombres de características con ejemplos específicos para EMG
    num_features = X_train.shape[1]
    feature_names = [
        "Frecuencia Media", 
        "Amplitud RMS", 
        "Frecuencia Mediana", 
        "Potencia Total", 
        "Varianza de Señal", 
        "Densidad de Potencia", 
        "Índice de Fatiga",
        "Frecuencia de Pico", 
        "Desviación Estándar", 
        "Valor de Cresta"
    ]

    # Ajustar la lista si el número de características es mayor o menor
    if num_features > len(feature_names):
        feature_names = feature_names * (num_features // len(feature_names)) + feature_names[:num_features % len(feature_names)]
    elif num_features < len(feature_names):
        feature_names = feature_names[:num_features]

    # Entrenar y evaluar modelo
    modelo = entrenar_y_evaluar_modelo(X_train, y_train)

    # Interpretación del modelo
    X_sample = shap.sample(X_test, 100)
    visualizacion_importancia_caracteristicas(modelo, X_sample, feature_names)

    # Conclusión final basada en la predicción
    conclusion_fatiga(modelo, X_test, y_test)

# Ejecutar el programa
if __name__ == '__main__':
    main()
