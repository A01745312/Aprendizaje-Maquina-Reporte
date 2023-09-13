# Paula Sophia Santoyo Arteaga
# A01745312
# 11-Sept-2023
# Uso de framework de aprendizaje máquina para la implementación de una solución
# ------------------------------------------------------------------------------


# Importar librerias
import matplotlib.pyplot as plt
from matplotlib_venn import venn3 as venn
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar dataset de wine de scikit-learn
def load_data():
    # Almacena el dataset de vinos
    wine = load_wine()
    # Almacena los datos de los atributos
    x = wine.data
    print(f'Datos de los atributos\n{x}\n{len(x)}')
    # Almacena los datos de la clasificación del vino
    y = wine.target
    print(f'Datos de la clasificación del vino\n{y}\n{len(y)}')
    # Almacena los nombres de la clasificacion de vinos (clase 0, 1 o 2)
    target_names = wine.target_names
    return x, y, target_names


def train_model(x, y, max_depth=None):
    # Dividir los datos en 2: entrenamiento y testing (temp)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, 
                                            test_size=0.3, random_state=42)
    # Dividir los datos para testing y validacións
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, 
                                            test_size=0.4, random_state=42)
    # Crear modelo de árbol de decisión con profundidad y 
    # semilla aleatoria personalizada
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    # Entrenar el modelo con los datos de entrenamiento (x_train y y_train)
    clf.fit(x_train, y_train)
    # Regresa los datos de entrenamiento, validación, testing 
    # y el modelo entrenado
    return x_train, x_val, x_test, y_train, y_val, y_test, clf

def evaluate_model(max_depth, clf, x, y, data_type):
    # Realiza predicciones de los datos de la variable x (train o test)
    y_pred = clf.predict(x)
    # Calcula la precisión del modelo
    accuracy = accuracy_score(y, y_pred)
    # Crea el informe de clasificación de los datos proporcionados
    classification = classification_report(y, y_pred, 
                        zero_division=0, target_names=target_names)
    # Crea la matriz de confusión
    confusion = confusion_matrix(y, y_pred)

    # Muestra el valor de precisión y el informe de clasificación del modelo
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{classification}')
    # Manda llamar la función que crea la gráfica de la matriz de confusión
    c_matrix(max_depth, data_type, confusion)
    return accuracy


def c_matrix(max_depth, data_type, matrix):
    # Genera la gráfica de la matriz de confusión
    plt.figure(figsize=(5, 4.5))
    sns.set(font_scale=1)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="RdPu", 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {data_type} Run {max_depth}')
    plt.show()


def prediction(clf):
    # Datos para hacer una predicción 
    new_wine = [[13.24, 2.59, 2.87, 21.0, 118.0, 2.8, 2.69, 
                 0.39, 1.82, 4.32, 1.04, 2.93, 735.0]]
    # Usar el modelo entrenado para hacer la predicción con los datos de arriba
    predictions = clf.predict(new_wine)
    # Muestra la predicción del vino ingresado
    print('\n~ PREDICCIÓN ~')
    print(f'Atributos del vino: {new_wine}')
    print(f'Predicción para el vino: {target_names[predictions[0]]}')

def bias_degree(clf, y_train, x_train, y_test, x_test, y_val, x_val):
    train_accuracy = accuracy_score(y_train, clf.predict(x_train))
    test_accuracy = accuracy_score(y_test, clf.predict(x_test))
    val_accuracy = accuracy_score(y_val, clf.predict(x_val))
    # Almacenar las precisiones en las listas
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    val_accuracies.append(val_accuracy)

def graph_bias():
    plt.figure(figsize=(8, 5))
    depths = list(range(1, 7))
    plt.plot(depths, train_accuracies, label='Entrenamiento', marker='o')
    plt.plot(depths, test_accuracies, label='Prueba', marker='o')
    plt.plot(depths, val_accuracies, label='Validación', marker='o')
    plt.title('Precisión vs Profundidad del Árbol de Decisión')
    plt.xlabel('Profundidad del Árbol')
    plt.ylabel('Precisión')
    plt.xticks(depths)
    plt.legend()
    plt.grid(True)
    plt.show()

def dataset_parts(x_train, x_test, x_val, y_train, y_test, y_val):
    # Imprimir partes de los datasets
    print('\nPartes del Conjunto de Entrenamiento (70%):')
    print(f'Número de muestras en el Conjunto de Entrenamiento: {len(x_train)}')
    print(f'X_train:\n{x_train}\n')
    print(f'y_train:\n{y_train}\n')

    print('\nPartes del Conjunto de Prueba (12%):')
    print(f'Número de muestras en el Conjunto de Prueba: {len(x_test)}')
    print(f'X_test:\n{x_test}\n')
    print(f'y_test:\n{y_test}\n')

    print('\nPartes del Conjunto de Validación (18%):')
    print(f'Número de muestras en el Conjunto de Validación: {len(x_val)}')
    print(f'X_val:\n{x_val}\n')
    print(f'y_val:\n{y_val}\n')
    
     # Crear un diagrama de Venn para mostrar la división de datos
    plt.figure(figsize=(8, 6))
    venn_diagram = venn(subsets=(len(x_train), len(x_test), len(x_val), 
                                  0, 0, 0, 
                                  0),
                         set_labels=('Entrenamiento', 'Prueba', 'Validación'))
    plt.title('División de Datos entre Conjuntos de Entrenamiento, Prueba y Validación')
    plt.show()
    
    
def learning_rate(clf, x_train, y_train, x_test, y_test, x_val, y_val):
    # Crea una gráfica mostrando la curva de aprendizaje entre los datos de entrenamiento y prueba
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x_train, y_train, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cv=5, scoring='accuracy', shuffle=True, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Calcula las curvas de aprendizaje para los datos de validación
    val_train_sizes, val_train_scores, val_test_scores = learning_curve(
        clf, x_val, y_val, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cv=5, scoring='accuracy', shuffle=True, random_state=42)

    val_train_scores_mean = np.mean(val_train_scores, axis=1)
    val_train_scores_std = np.std(val_train_scores, axis=1)
    val_test_scores_mean = np.mean(val_test_scores, axis=1)
    val_test_scores_std = np.std(val_test_scores, axis=1)

    # Crear la gráfica de la curva de aprendizaje
    plt.figure(figsize=(8, 5))
    plt.title(f'Curva de Aprendizaje (Max Depth = {max_depth})')
    plt.xlabel('Tamaño del Conjunto de Entrenamiento')
    plt.ylabel('Precisión')
    plt.grid(True)

    # Rellenar el área entre las medias ± desviaciones estándar de las puntuaciones para el conjunto de entrenamiento
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')

    # Rellenar el área entre las medias ± desviaciones estándar de las puntuaciones para el conjunto de prueba
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    # Rellenar el área entre las medias ± desviaciones estándar de las puntuaciones para el conjunto de validación
    plt.fill_between(val_train_sizes, val_test_scores_mean - val_test_scores_std,
                     val_test_scores_mean + val_test_scores_std, alpha=0.1, color='y')

    # Dibujar las medias de las puntuaciones para el conjunto de entrenamiento
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Entrenamiento')

    # Dibujar las medias de las puntuaciones para el conjunto de prueba
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Prueba')

    # Dibujar las medias de las puntuaciones para el conjunto de validación
    plt.plot(val_train_sizes, val_test_scores_mean, 'o-', color='y', label='Validación')

    # Mostrar leyenda
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # Llama la función para obtener los datos
    x, y, target_names = load_data()
    
    train_accuracies = []
    test_accuracies = []
    val_accuracies = []
    
    # Ejecuta el entrenamiento, validación y test de los datos 3 veces
    for max_depth in range(1, 7):
        # Obtiene los datos para entrenamiento, validación, test y el modelo entrenado
        x_train, x_val, x_test, y_train, y_val, y_test, clf = train_model(x, 
                                                                    y, max_depth)
        # Muestra la ejecución que está corriendo como la profundidad máxima con la que 
        # se está entrenando
        print(f'\n*****   RUN {max_depth}   *****\nMax Depth: {max_depth}\n')
        print('                      ~ TRAINING DATA ~\n')
        # Se evalua el modelo con los datos de entrenamiento
        evaluate_model(max_depth, clf, x_train, y_train, "Training")
        print('\n                      ~ TEST DATA ~')
        # Se evalua el modelo con los datos de testing
        evaluate_model(max_depth, clf, x_test, y_test, "Test")
        print('\n                      ~ VALIDATION DATA ~')
        # Se evalúa el modelo con los datos de validación
        evaluate_model(max_depth, clf, x_val, y_val, "Validation")
        print('-' * 40)
        # Hace una gráfica con los valores de precisión de cada uno de los datos
        bias_degree(clf, y_train, x_train, y_test, x_test, y_val, x_val)
        learning_rate(clf, x_train, y_train, x_test, y_test, x_val, y_val)
    
    # Llama la función que hace la predicción con los datos de un vino nuevo
    prediction(clf)
    graph_bias()
    dataset_parts(x_train, x_test, x_val, y_train, y_test, y_val)