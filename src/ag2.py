import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def potencia_segura(x, p):
    x = np.clip(x, 1e-10, np.inf)
    return np.power(x, p)

class Cromosoma:
    def __init__(self, num_caracteristicas):
        self.pesos = np.random.uniform(-10, 10, num_caracteristicas)
        self.exponentes = np.random.uniform(-10, 10, num_caracteristicas)
        self.constante = np.random.uniform(-100, 100)
        self.puntuacion = None

    def calcular_aptitud(self, X, y):
        predicciones = np.sum([self.pesos[i] * potencia_segura(X[:, i], self.exponentes[i]) for i in range(X.shape[1])], axis=0) + self.constante
        self.puntuacion = mean_squared_error(y, predicciones)
        return self.puntuacion

    def __repr__(self):
        return f"Cromosoma(pesos={self.pesos}, exponentes={self.exponentes}, constante={self.constante}, puntuación={self.puntuacion})"

class AG:
    def __init__(self, datos_train, datos_test, seed=123, nInd=80, maxIter=120):
        self.archivo_entrenamiento = datos_train
        self.archivo_prueba = datos_test
        random.seed(seed)
        np.random.seed(seed)
        
        self.cargar_datos()
        self.poblacion = self.inicializar_poblacion(nInd, self.X_entrenamiento.shape[1])
        self.numGeneraciones = maxIter
        self.tamPoblacion = nInd

    def cargar_datos(self):
        try:
            datos_entrenamiento = pd.read_csv(self.archivo_entrenamiento)
            datos_prueba = pd.read_csv(self.archivo_prueba)
        except Exception as e:
            raise ValueError(f"Error al cargar los archivos de datos: {e}")
        
        escalador = MinMaxScaler()
        self.X_entrenamiento = escalador.fit_transform(datos_entrenamiento.drop('y', axis=1))
        self.y_entrenamiento = datos_entrenamiento['y'].values
        self.X_prueba = escalador.transform(datos_prueba.drop('y', axis=1))
        self.y_prueba = datos_prueba['y'].values

    def inicializar_poblacion(self, tamano, num_caracteristicas):
        return [Cromosoma(num_caracteristicas) for _ in range(tamano)]

    def elegir_progenitor(self):
        aptitud_total = sum(1 / (ind.puntuacion + 1e-6) for ind in self.poblacion)
        punto_seleccion = random.uniform(0, aptitud_total)
        acumulado = 0
        for individuo in self.poblacion:
            acumulado += 1 / (individuo.puntuacion + 1e-6)
            if acumulado > punto_seleccion:
                return individuo

    def cruzar(self, padre_a, padre_b):
        if random.random() < 0.2:
            return (padre_a, padre_b) if padre_a.puntuacion < padre_b.puntuacion else (padre_b, padre_a)

        punto_cruce = random.randint(1, len(padre_a.pesos) - 1)
        hijo_a = Cromosoma(len(padre_a.pesos))
        hijo_b = Cromosoma(len(padre_b.pesos))

        hijo_a.pesos = np.concatenate((padre_a.pesos[:punto_cruce], padre_b.pesos[punto_cruce:]))
        hijo_a.exponentes = np.concatenate((padre_a.exponentes[:punto_cruce], padre_b.exponentes[punto_cruce:]))
        hijo_a.constante = padre_a.constante if random.random() > 0.5 else padre_b.constante

        hijo_b.pesos = np.concatenate((padre_b.pesos[:punto_cruce], padre_a.pesos[punto_cruce:]))
        hijo_b.exponentes = np.concatenate((padre_b.exponentes[:punto_cruce], padre_a.exponentes[punto_cruce:]))
        hijo_b.constante = padre_b.constante if random.random() > 0.5 else padre_a.constante

        return hijo_a, hijo_b

    def aplicar_mutacion(self, individuo, prob_mutacion=0.0215):
        for i in range(len(individuo.pesos)):
            if random.random() < prob_mutacion:
                individuo.pesos[i] = np.random.uniform(-10, 10)
                individuo.exponentes[i] = np.random.uniform(-10, 10)
        if random.random() < prob_mutacion:
            individuo.constante = np.random.uniform(-100, 100)

    def run(self):
        for individuo in self.poblacion:
            individuo.calcular_aptitud(self.X_entrenamiento, self.y_entrenamiento)

        for generacion in range(self.numGeneraciones):
            nueva_poblacion = []

            for _ in range(self.tamPoblacion // 2):
                progenitor1 = self.elegir_progenitor()
                progenitor2 = self.elegir_progenitor()
                descendencia1, descendencia2 = self.cruzar(progenitor1, progenitor2)
                self.aplicar_mutacion(descendencia1)
                self.aplicar_mutacion(descendencia2)
                nueva_poblacion.extend([descendencia1, descendencia2])

            for individuo in nueva_poblacion:
                individuo.calcular_aptitud(self.X_entrenamiento, self.y_entrenamiento)

            self.poblacion = nueva_poblacion

            mejor_individuo = min(self.poblacion, key=lambda ind: ind.puntuacion)
            print(f"Generación {generacion}: Mejor aptitud = {mejor_individuo.puntuacion}")

        mejor_individuo = min(self.poblacion, key=lambda ind: ind.puntuacion)
        predicciones = np.sum([mejor_individuo.pesos[i] * potencia_segura(self.X_prueba[:, i], mejor_individuo.exponentes[i]) for i in range(self.X_prueba.shape[1])], axis=0) + mejor_individuo.constante

        return mejor_individuo, predicciones
