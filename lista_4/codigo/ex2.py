import matplotlib.pyplot as plt
import numpy as np

from ga import IndividualAbstract, Population

if __name__ == "__main__":

    class Individual(IndividualAbstract):

        def func_to_eval(self):
            pass

        def fitness(self):
            if self.type_gen == 'binary':
                self.score = sum(self.gen)
            return self.score

    type_gen = 'binary'
    type_ps = 'sus'
    tp_xover = 'one_pt'
    tp_mut = 'binary'
    num_alelo = 25
    xover_rate = 0.7
    mutation_rate = 1 / num_alelo
    pop_size = 100
    num_parents = 100
    num_xover = 50
    num_gen = 10

    best_x = []
    geracao = 0

    pop = Population(Typ=Individual, type_gen=type_gen, num_alelo=num_alelo, pop_size=pop_size)
    pop_fitness = pop.fitness()
    best_x.append(min(pop_fitness))

    for n in range(num_gen):
        pop.parent_selection(type_ps=type_ps, num_parents=num_parents)
        pop.eval_xover(tp_xover=tp_xover, xover_rate=xover_rate, num_xover=num_xover)
        pop.mutation(tp_mut=tp_mut, mutation_rate=mutation_rate)
        pop.individuals = pop.mating_pool[num_parents:]
        offspring_fitness = pop.fitness()
        best_x.appenad(min(offspring_fitness))

    print(
        f'O mínimo {min(best_x):.5} foi encontrado na geração de número {best_x.index(min(best_x))}.'
    )

    SHOW_PLOT = False
    # Plot fitness history
    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(best_x)), best_x)
        plt.ylabel('Fitness')
        plt.xlabel('Generations')
        plt.show()
