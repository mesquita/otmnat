import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ga import IndividualAbstract, Population

if __name__ == "__main__":

    class Individual(IndividualAbstract):
        def fitness(self):
            x = np.array(self.gen)
            n = len(x)
            self.score = -20 * np.exp(-0.2 * np.sqrt(1 / n * np.sum(x**2))) - np.exp(
                1 / n * np.sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
            return self.score

    type_gen = 'float'
    type_ps = 'sus'
    tp_xover = 'gbl_dscrt'
    tp_mut = 'float'
    needs_sigma = 'yes'
    tp_selection = 'ranking'
    num_alelo = 30
    xover_rate = 0.7
    mutation_rate = 1 / num_alelo
    pop_size = 30
    num_xover = 200
    num_gen = 1000

    best_x = []
    geracao = 0

    pop = Population(
        Typ=Individual,
        type_gen=type_gen,
        needs_sigma=needs_sigma,
        num_alelo=num_alelo,
        pop_size=pop_size)
    pop_fitness = pop.fitness()
    best_x.append(min(pop_fitness))

    for n in tqdm(range(num_gen)):
        pop.parent_selection()
        pop.eval_xover(tp_xover=tp_xover, xover_rate=xover_rate, num_xover=num_xover)
        pop.mutation(tp_mut=tp_mut, mutation_rate=mutation_rate)
        pop.individuals = pop.suvivor_selection(tp_selection=tp_selection)
        offspring_fitness = pop.fitness()
        best_x.append(min(offspring_fitness))

    print(
        f'O mínimo {min(best_x)} foi encontrado na geração de número {best_x.index(min(best_x))}.')

    SHOW_PLOT = True
    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(best_x)), best_x)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.show()
