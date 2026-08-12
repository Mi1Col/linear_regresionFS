# TODO:
# - Hacer documentacion:
#   - README.md
#   - Documentacion_matematica.pdf (no se llamará asi)

import matplotlib.pyplot as plt
import random
import time

w = 0 # Weight / Peso
B = 0 # Bias / Sesgo

puntos = [[],[]]
def crearPuntos(max_length, m, b, variacion):
    # Voy a crear diferentes puntos que caigan alrdedor de una recta con una desviacion que podamos determinar
    # El formato será el [[x0, x1, x2, ... xn], [y0, y1, y2, ... yn]]


    if variacion != 0: # Si hay una variacion
        max_variacion = (100 + variacion) * 100 # Calculamos la variacion maxima
        min_variacion = (100 - variacion) * 100 # Calculamos la variacion minima
        for i in range(max_length):
            puntos[0].append(i)
            y = m*i + b # Caculamos la y REAL
            y *= (random.randrange(min_variacion, max_variacion) / 10000) # La multiplicamos por una variacion aleatorioa entre los limites determinados
            y = round(y, 2) # Redondeamos la y REAL a 2 decimales
            puntos[1].append(y) # Ponemos la y REAL a la lista de y REAL
    if variacion == 0: # Si no hay variacion hace el calculo normal, sin variacion
        for i in range(max_length):
            puntos[0].append(i)
            y = m*i + b # Caculamos la y REAL
            y = round(y, 2) # Redondeamos la y REAL a 2 decimales
            puntos[1].append(y) # Ponemos la y REAL a la lista de y REAL


    plt.ion()
    fig = plt.figure(figsize=(10, 4)) # Esto determina la relacion de aspecto de la grafica (pero tambien el tamaño, aunque eso no se como lo determina, si tu pones 1:1 es un cuadrado mas pequeño que si pones 4:4)
    dot = plt.scatter(puntos[0], puntos[1], color="#7D7D7D") # Esto es para dibujar los puntos REALES
    linea, = plt.plot(puntos[0], [m * x + b for x in puntos[0]], color="#00ff00") # Esto es para dibujar la linea REAL
    linea_neurona, = plt.plot(
    puntos[0], # Eje x
    [w * x + B for x in puntos[0]], # Eje y de PREDICCIONES
    color="red"
) # Esto es para dibujar la linea predicha por la neurona
    plt.xticks(range(0, max(puntos[0]), 1), ) # Esto determina el inicio, fin y paso del eje x de la grafica
    plt.show(block=False) # Muetsra el grafico
    plt.pause(0.1)

    return fig, linea, linea_neurona

fig, linea, linea_neurona = crearPuntos(15, 0.2, 3, 0)

# Ahora voy a intentar crear una neurona desde cero que sea capaz de encontrar la ecuacion de esa recta basado en los puntos

# El entrenamiento será asi:
# Input -> Peso * Input + Bias -> Error -> Ajuste

puntoX = puntos[0]
puntoY = puntos[1]
c = 0
tasa = 0.001

# Aqui entrenamos a la neurona de una manera basatnte rudimentaria
""" while True:
    print("Loop: ", c)
    c += 1

    error_total = 0

    for i in range(len(puntoX)):

        entrada = puntoX[i]
        esperado = puntoY[i]

        resultado = entrada * w + B

        error = esperado - resultado

        w += tasa * error * entrada
        B += tasa * error

        error_total = abs(error)
    error_medio = error_total / len(puntoX)

    if c % 10 == 0:
        print(w, B, error_medio)

    if error_medio < 0.001:
        break
 """


inicio = time.perf_counter()
# Voy a intentar entrenarla usando el descenso de gradiente que es lo que se suele usar.
# El entrenamiento aqui seria parecido:
# Input -> PREDICCIONES de y -> MSE -> Gradientes (Derivadas parciales, mirar documentacion) -> Descenso de gradiente (Ajuste)
contador = 0
while True:
    predicciones = [] # Crea la lista donde se guardaran las predicciones hechas en cada Loop / Epoch
    contador +=1 # Esto es solo para llevar una cuenta de los Loop / Epoch

    # Calculamos las predicciones
    for i in range(len(puntoX)):
        entrada = puntoX[i]

        prediccion = entrada * w + B
        predicciones.append(prediccion) # Agregamos la prediccion hecha en esta vuelta a la lista de predicciones asi hasat quedarnos sin entradas osea, puntos X

    # Ahora calculamos el error de todas las predicciones
    errores = []
    for i in range(len(predicciones)):
        error = predicciones[i] - puntoY[i] # Calcula el error real
        errores.append(error) # Agrega el error real a la lisya de errores asi hasat quedarnos sin predicciones

    # Ahora calculamos el MSE (Error Cuadratico Medio)
    # 1. Elevamos todos los errores al cuadrado
    erroresCuadrado = []
    for i in range(len(errores)):
        erroresCuadrado.append(errores[i]**2)

    # 2. Sumamos todos los errores al cuadrado y dividimos entre el total de elementos para obtner el ERROR CUADRATICO MEDIO (MSE)
    MSE = sum(erroresCuadrado) / len(erroresCuadrado)

    # Ahora necesito calcular como cambia MSE respecto a w y B.
    # Esta es la parte mas complicada porque entran derivadas paciales que estarán explicadas en la documentacion
    error_por_entrada = [] # En la documentacion se explica porque necesito esto. Resumidamente: necesito tener (predicciones - real) * puntoX
    for i in range(len(puntoX)):
        error_por_entrada.append(errores[i] * puntoX[i])
    grad_w = (2/len(puntoX)) * sum(error_por_entrada) # Derivada parcial, mira documentacion
    grad_B = (2/len(puntoX)) * sum(errores) # Derivada parcial, mira documentacion

    w -= tasa * grad_w # Es una resta porque necesitamos ir en la direccion contraria a la pendiente, que es donde el error es minimo
    B -= tasa * grad_B # Lo mismo que arriba

    # print("Loop:", contador, w, B, MSE ** 0.5)

    if MSE ** 0.5 < 0.00001: # Si el error medio real es menor que 0.00001 me conformo con esa accuracy y cierro el entrenamiento
        break

    if contador % 500 == 0: # Por cada 500 vueltas a este while se actualiza la linea dibujada por la neurona
        linea_neurona.set_ydata([w * x + B for x in puntoX])
        plt.pause(0.001)
fin = time.perf_counter()
print(f"\nw: {w:.1f}\tb: {round(B, 3)}\tRMSE: {MSE ** 0.5}\n\nTime: {fin-inicio:.2f} s\tEpcoh: {contador}\tEpoch/s: {contador/(fin-inicio):.2f}")

def probar_neruona(x: list):
    # print("\nHaz una prediccion!\nIntroduce un numero x y se te dará un valor y usando la neurona ya entrenada.\n")
    for i in x: # Recorrera la lista x elemento por elemento
        y = w*i+B # y es el numero que predice la neurona dado una x
        print(f"[{i}, {y}]") 
        plt.scatter(i, y, c="#5500ff") # Esto es para que se dibuje un nuevo punto para que sea la x y la prediccion que toca
        if i >= 0: # Este if else es para que si la x es negativa tambien se expanda hacia los numero negativos correctamente
            plt.xticks(range(0, i + 20, 5))
        else:
            plt.xticks(range(0, i - 20, 5))
        puntoX.append(i) # Agrega el punto x a predecir a la lista de puntoX.
        linea_neurona.set_xdata(puntoX) # Actualiza los valores de x
        linea_neurona.set_ydata([w * j + B for j in puntoX]) # Actualiza los valores de y
        plt.show()

# probar_neruona(range(30, 101, 10))

while plt.fignum_exists(fig.number):
    plt.pause(0.1)

# LO HE LOGRADOOOOO!!!!
# SE SIENTE TAN BIEN
# HE ENTRENADO MI PRIMERA NEURONA ARTIFICIAL
