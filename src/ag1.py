import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error

class Chromosome:
    def __init__(self, num_features):
        self.weights = np.random.uniform(-10, 10, num_features)
        self.powers = np.random.uniform(-10, 10, num_features)
        self.bias = np.random.uniform(-100, 100)
        self.score = None

    def compute_fitness(self, X, y):
        predictions = np.sum([self.weights[i] * X[:, i]**self.powers[i] for i in range(X.shape[1])], axis=0) + self.bias
        self.score = mean_squared_error(y, predictions)
        return self.score

    def __repr__(self):
        return f"Chromosome(weights={self.weights}, powers={self.powers}, bias={self.bias}, score={self.score})"

class AG:
    def __init__(self, train_file, test_file, seed=123, population_size=50, generations=100):
        self.train_file = train_file
        self.test_file = test_file
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        random.seed(seed)
        np.random.seed(seed)
        
        # Cargar datos
        self.training_data = pd.read_csv(train_file)
        self.validation_data = pd.read_csv(test_file)
        
        # Separar características y etiquetas
        self.X_train = self.training_data.drop('y', axis=1).values
        self.y_train = self.training_data['y'].values
        self.X_test = self.validation_data.drop('y', axis=1).values
        self.y_test = self.validation_data['y'].values
        
        self.population = self.initialize_population(population_size, self.X_train.shape[1])

    def initialize_population(self, size, num_features):
        return [Chromosome(num_features) for _ in range(size)]

    def select_parent(self, population):
        total_fitness = sum(ind.score for ind in population)
        selection_point = random.uniform(0, total_fitness)
        current = 0
        for individual in population:
            current += individual.score
            if current > selection_point:
                return individual

    def perform_crossover(self, parent_a, parent_b):
        offspring_a = Chromosome(len(parent_a.weights))
        offspring_b = Chromosome(len(parent_a.weights))
        crossover_point = random.randint(1, len(parent_a.weights) - 1)

        offspring_a.weights = np.concatenate((parent_a.weights[:crossover_point], parent_b.weights[crossover_point:]))
        offspring_a.powers = np.concatenate((parent_a.powers[:crossover_point], parent_b.powers[crossover_point:]))
        offspring_a.bias = parent_a.bias if random.random() > 0.5 else parent_b.bias

        offspring_b.weights = np.concatenate((parent_b.weights[:crossover_point], parent_a.weights[crossover_point:]))
        offspring_b.powers = np.concatenate((parent_b.powers[:crossover_point], parent_a.powers[crossover_point:]))
        offspring_b.bias = parent_b.bias if random.random() > 0.5 else parent_a.bias

        return offspring_a, offspring_b

    def apply_mutation(self, individual, mutation_prob=0.01):
        for i in range(len(individual.weights)):
            if random.random() < mutation_prob:
                individual.weights[i] = np.random.uniform(-10, 10)
                individual.powers[i] = np.random.uniform(-10, 10)
        if random.random() < mutation_prob:
            individual.bias = np.random.uniform(-100, 100)

    def execute(self):
        for individual in self.population:
            individual.compute_fitness(self.X_train, self.y_train)

        for generation in range(self.generations):
            new_population = []

            # Selección y reproducción
            for _ in range(self.population_size // 2):
                parent1 = self.select_parent(self.population)
                parent2 = self.select_parent(self.population)
                child1, child2 = self.perform_crossover(parent1, parent2)
                self.apply_mutation(child1)
                self.apply_mutation(child2)
                new_population.extend([child1, child2])

            # Evaluar nueva población
            for individual in new_population:
                individual.compute_fitness(self.X_train, self.y_train)

            # Reemplazar la población anterior con la nueva
            self.population = new_population

            # Imprimir el mejor individuo de la generación actual
            best_individual = min(self.population, key=lambda ind: ind.score)
            print(f"Generation {generation}: Best Fitness = {best_individual.score}")

        # Devolver el mejor individuo encontrado
        best_individual = min(self.population, key=lambda ind: ind.score)

        # Predicciones sobre el conjunto de test
        y_predictions = np.sum([best_individual.weights[i] * self.X_test[:, i]**best_individual.powers[i] for i in range(self.X_test.shape[1])], axis=0) + best_individual.bias

        return best_individual, y_predictions

# Ejemplo de ejecución
# ga = GeneticAlgorithm('datos_train.csv', 'datos_test.csv')
# best_ind, preds = ga.execute()
# print(best_ind)
# print(preds)
