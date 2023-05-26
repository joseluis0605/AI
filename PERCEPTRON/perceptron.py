import numpy as np

# Definición de los datos de entrada
X = np.array([[1, 0, 1, 0, 0, 0],
              [1, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0],
              [1, 1, 0, 0, 1, 1], #A4
              [1, 1, 1, 1, 0, 0],  # A5
              [1, 0, 0, 0, 1, 1],  # A6
              [1, 0, 0, 0, 1, 0],  # A7
              [0, 1, 1, 1, 0, 1],  # A8
              [0, 1, 1, 0, 1, 1],  # A9
              [0, 0, 0, 1, 1, 0],  # A10
              [0, 1, 0, 1, 0, 1],  # A11
              [0, 0, 0, 1, 0, 1],  # A12
              [0, 1, 1, 0, 1, 1],  # A13
              [0, 1, 1, 1, 0, 0]]) # A14

# Definición de las salidas objetivo
T = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

# Inicializar los pesos con ceros
w = np.array([0] * len(X[0]))

# Definir el umbral
b = 1

# Definir la tasa de aprendizaje
alfa = 1


# Definir la función de activación
def activation_function(x):
    return 1 if x >= 0 else 0

# Iterar hasta que todos los ejemplos sean clasificados correctamente
j = 0
while True:
    misclassified = False
    for i in range(X.shape[0]):
        x = X[i]
        t = T[i]

        # Calcular el valor de activación
        a = np.dot(w, x) + b

        # Aplicar la función de activación
        y = activation_function(a)

        # Actualizar los pesos si el ejemplo es clasificado incorrectamente
        if y != t:
            w += alfa * (t - y) * x
            b += alfa * (t - y)
            misclassified = True


    j += 1
    print("Epoca: ", j)
    print("Pesos : ", w)
    print("Umbral : ", b)


    # Salir del bucle si todos los ejemplos son clasificados correctamente
    if not misclassified:
        break

# Imprimir los pesos finales
print("################")
print("################")
print("################")

print("SOLUCIONES:")
print("Pesos finales: ", w)
print("Umbral final: ", b)
print("Epocas: ", j)