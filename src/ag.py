import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error

class Individual:
    def __init__(self, num_attributes):
        self.coefficients = np.random.uniform(-10, 10, num_attributes)
        self.exponents = np.random.uniform(-10, 10, num_attributes)
        self.constant = np.random.uniform(-100, 100)
        self.fitness = None

    def evaluate(self, X, y):
        predictions = np.sum([self.coefficients[i] * X[:, i]**self.exponents[i] for i in range(X.shape[1])], axis=0) + self.constant
        self.fitness = mean_squared_error(y, predictions)
        return self.fitness

    def __repr__(self):
        return f"Individual(coefficients={self.coefficients}, exponents={self.exponents}, constant={self.constant}, fitness={self.fitness})"

class AG:
    def __init__(self, datos_train, datos_test, seed=123, nInd=50, maxIter=100):
        self.datos_train = datos_train
        self.datos_test = datos_test
        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter
        random.seed(seed)
        np.random.seed(seed)
        
        # Cargar datos
        self.train_data = pd.read_csv(datos_train)
        self.test_data = pd.read_csv(datos_test)
        
        # Separar características y etiquetas
        self.X_train = self.train_data.drop('y', axis=1).values
        self.y_train = self.train_data['y'].values
        self.X_test = self.test_data.drop('y', axis=1).values
        self.y_test = self.test_data['y'].values
        
        self.population = self.create_initial_population(nInd, self.X_train.shape[1])

    def create_initial_population(self, pop_size, num_attributes):
        return [Individual(num_attributes) for _ in range(pop_size)]

    def roulette_selection(self, population):
        max_fitness = sum(ind.fitness for ind in population)
        pick = random.uniform(0, max_fitness)
        current = 0
        for ind in population:
            current += ind.fitness
            if current > pick:
                return ind

    def crossover(self, parent1, parent2):
        child1 = Individual(len(parent1.coefficients))
        child2 = Individual(len(parent1.coefficients))
        cross_point = random.randint(1, len(parent1.coefficients) - 1)

        child1.coefficients = np.concatenate((parent1.coefficients[:cross_point], parent2.coefficients[cross_point:]))
        child1.exponents = np.concatenate((parent1.exponents[:cross_point], parent2.exponents[cross_point:]))
        child1.constant = parent1.constant if random.random() > 0.5 else parent2.constant

        child2.coefficients = np.concatenate((parent2.coefficients[:cross_point], parent1.coefficients[cross_point:]))
        child2.exponents = np.concatenate((parent2.exponents[:cross_point], parent1.exponents[cross_point:]))
        child2.constant = parent2.constant if random.random() > 0.5 else parent1.constant

        return child1, child2

    def mutate(self, individual, mutation_rate=0.01):
        for i in range(len(individual.coefficients)):
            if random.random() < mutation_rate:
                individual.coefficients[i] = np.random.uniform(-10, 10)
                individual.exponents[i] = np.random.uniform(-10, 10)
        if random.random() < mutation_rate:
            individual.constant = np.random.uniform(-100, 100)

    def run(self):
        for individual in self.population:
            individual.evaluate(self.X_train, self.y_train)

        for generation in range(self.maxIter):
            new_population = []

            # Selection and reproduction
            for _ in range(self.nInd // 2):
                parent1 = self.roulette_selection(self.population)
                parent2 = self.roulette_selection(self.population)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.extend([child1, child2])

            # Evaluate new population
            for individual in new_population:
                individual.evaluate(self.X_train, self.y_train)

            # Replace old population with new population
            self.population = new_population

            # Print the best individual of the current generation
            best_individual = min(self.population, key=lambda ind: ind.fitness)
            print(f"Generation {generation}: Best Fitness = {best_individual.fitness}")

        # Return the best individual found
        best_individual = min(self.population, key=lambda ind: ind.fitness)

        # Predicciones sobre el conjunto de test
        y_pred = np.sum([best_individual.coefficients[i] * self.X_test[:, i]**best_individual.exponents[i] for i in range(self.X_test.shape[1])], axis=0) + best_individual.constant

        return best_individual, y_pred
