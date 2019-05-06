from abc import ABC, abstractmethod

import numpy as np


class IndividualAbstract(ABC):
    def __init__(self, type_gen='binary', num_alelo=5, given_alleles=None):
        self.type_gen = type_gen
        self.num_alelo = num_alelo

        if given_alleles is None:
            if self.type_gen == 'float':
                self.gen = [b for b in np.random.uniform(low=-1, high=1, size=self.num_alelo)]
            else:
                self.gen = [int(bool(b)) for b in np.random.randint(2, size=self.num_alelo)]
        else:
            self.gen = given_alleles

        super(IndividualAbstract, self).__init__()

    @abstractmethod
    def func_to_eval(self):
        pass

    @abstractmethod
    def fitness(self):
        pass


class Population(object):
    def __init__(self, Typ=None, type_gen='binary', num_alelo=5, pop_size=10):
        self.type_gen = type_gen
        self.pop_size = pop_size
        self.num_alelo = num_alelo
        self.Typ = Typ

        if self.Typ is None:
            class IndividualGenerico(IndividualAbstract):
                def func_to_eval(self):
                    x = self.phenotype
                    return x**2 - 0.3 * np.cos(10 * np.pi * x)

                def fitness(self):
                    if self.type_gen == 'binary':
                        b = [self.gen[-1]]
                        for n in self.gen[-2::-1]:
                            b.insert(0, b[0] ^ n)
                        u = sum(2**k for k in range(self.num_alelo) if b[k])
                        u /= 2**self.num_alelo
                        self.phenotype = -2 + 4 * u
                        self.score = self.func_to_eval()
                    return self.score
            self.Typ = self.IndividualGenerico

        self.individuals = []
        for x in range(self.pop_size):
            self.individuals.append(self.Typ(type_gen=self.type_gen, num_alelo=self.num_alelo))

    def fitness(self):
        self.score = []
        for indv in self.individuals:
            self.score.append(indv.fitness())
        return self.score

    def parent_selection(self, type_ps='sus', shift=None, num_parents=None):
        if num_parents is None:
            num_parents = self.pop_size

        if type_ps == 'sus':
            if shift is None:
                shifted_fitness = self.score
            else:
                shifted_fitness = self.score
                shifted_fitness[:] = [shift - x for x in self.score]
            pre_a = (shifted_fitness / sum(shifted_fitness))
            a = np.cumsum(pre_a)
            self.mating_pool = []
            current_member = 1
            ue = 1
            r = np.random.uniform(0, 1 / num_parents)
            while (current_member <= num_parents):
                while (r <= a[ue]):
                    self.mating_pool.append(self.individuals[ue])
                    r = r + 1 / num_parents
                    current_member = current_member + 1
                ue = ue + 1
            return self.mating_pool

    def xover(self, tp_xover='one_pt', xover_rate=0.7):
        if tp_xover == 'one_pt':
            i, j = np.random.choice(range(len(self.mating_pool)), 2)
            if np.random.uniform(0, 1) < xover_rate:
                xchg_pt = np.random.choice(range(self.num_alelo))
                self.mating_pool.append(
                    self.Typ(
                        type_gen=self.type_gen,
                        num_alelo=self.num_alelo,
                        given_alleles=self.mating_pool[i].gen[:xchg_pt] +
                        self.mating_pool[j].gen[xchg_pt:]))
                self.mating_pool.append(
                    self.Typ(
                        type_gen=self.type_gen,
                        num_alelo=self.num_alelo,
                        given_alleles=self.mating_pool[j].gen[:xchg_pt] +
                        self.mating_pool[i].gen[xchg_pt:]))

    def eval_xover(self, tp_xover='one_pt', xover_rate=0.7, num_xover=None):
        if num_xover is None:
            num_xover = self.num_parents
        for i in range(num_xover):
            self.xover(tp_xover=tp_xover, xover_rate=xover_rate)

    def mutation(self, tp_mut='binary', mutation_rate=0.2):
        if tp_mut == 'binary':
            for parent in self.mating_pool:
                for allel in parent.gen:
                    if np.random.uniform(0, 1) < mutation_rate:
                        allel = int(not (allel))
