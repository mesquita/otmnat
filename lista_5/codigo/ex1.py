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
    tp_mut = 'metaEP'
    needs_sigma = 'yes'
    tp_selection = 'tour'
    num_alelo = 4
    xover_rate = 0.7
    pop_size = 200
    num_gen = 200

    best_x = []

    pop = Population(
        Typ=Individual,
        type_gen=type_gen,
        needs_sigma=needs_sigma,
        num_alelo=num_alelo,
        pop_size=pop_size)
    best_x.append(min(pop.fitness()))

    for n in tqdm(range(num_gen)):
        pop.parent_selection()
        pop.mutation(tp_mut=tp_mut)
        pop.individuals = pop.suvivor_selection(tp_selection=tp_selection, num_rounds=10)
        best_x.append(min(pop.fitness()))

    print(
        f'O mínimo {min(best_x)} foi encontrado na geração de número {best_x.index(min(best_x)) + 1}.'
    )

    SHOW_PLOT = True
    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(best_x)), best_x)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.show()
