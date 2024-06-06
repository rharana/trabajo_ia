import time
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from ag import AG

# Nombre genérico del dataset
nombre_dataset = 'toy1'

nombre_dataset_train = f"../data/{nombre_dataset}_train.csv"
nombre_dataset_val = f"../data/{nombre_dataset}_val.csv"

# Crear instancia del AG
ag = AG(
    datos_train=nombre_dataset_train, 
    datos_test=nombre_dataset_val, 
    seed=123, 
    nInd=50, 
    maxIter=100
)

# Ejecución del AG midiendo el tiempo
inicio = time.time()
ind, y_pred = ag.run()
fin = time.time()
print(f'Tiempo ejecución: {(fin - inicio):.2f} segundos')

# Imprimir mejor solución encontrada
print(f'Mejor individuo: {ind}')

# Imprimir predicciones sobre el conjunto de test
print(f'Predicciones: {y_pred}')

# Cargar valores reales de 'y' en el conjunto de validación/test 
# y calcular RMSE y R2 con las predicciones del AG
y_true = pd.read_csv(nombre_dataset_val)['y']
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f'RMSE: {rmse:.4f}')

r2 = r2_score(y_true, y_pred)
print(f'R2: {r2:.4f}')