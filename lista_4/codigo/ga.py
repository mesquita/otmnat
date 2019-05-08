from abc import ABC, abstractmethod

import numpy as np


class IndividualAbstract(ABC):
    def __init__(self, type_gen='binary', needs_sigma='no', num_alelo=5, given_alleles=None):
        self.type_gen = type_gen
        self.num_alelo = num_alelo
        self.needs_sigma = needs_sigma

        if given_alleles is None:
            if self.type_gen == 'float':
                self.gen = [b for b in np.random.uniform(low=-1, high=1, size=self.num_alelo)]
            else:
                self.gen = [int(bool(b)) for b in np.random.randint(2, size=self.num_alelo)]
        else:
            self.gen = given_alleles
        if self.needs_sigma == 'yes':
            self.sigma = [b for b in np.random.normal(loc=0.0, scale=1.0, size=self.num_alelo)]
        super(IndividualAbstract, self).__init__()

    @abstractmethod
    def fitness(self):
        pass


class Population(object):
    def __init__(self, Typ=None, type_gen='binary', needs_sigma='no', num_alelo=5, pop_size=10):
        self.type_gen = type_gen
        self.pop_size = pop_size
        self.num_alelo = num_alelo
        self.Typ = Typ
        self.needs_sigma = needs_sigma

        if self.Typ is None:

            class IndividualGenerico(IndividualAbstract):
                def fitness(self):
                    if self.type_gen == 'binary':
                        b = [self.gen[-1]]
                        for n in self.gen[-2::-1]:
                            b.insert(0, b[0] ^ n)
                        u = sum(2**k for k in range(self.num_alelo) if b[k])
                        u /= 2**self.num_alelo
                        self.phenotype = -2 + 4 * u
                        x = self.phenotype
                        self.score = x**2 - 0.3 * np.cos(10 * np.pi * x)
                    return self.score

            self.Typ = self.IndividualGenerico

        self.individuals = []
        for x in range(self.pop_size):
            self.individuals.append(
                self.Typ(
                    type_gen=self.type_gen, needs_sigma=self.needs_sigma, num_alelo=self.num_alelo))

    def fitness(self, whose='population'):
        if whose == 'population':
            self.score = []
            for indv in self.individuals:
                self.score.append(indv.fitness())
            return self.score
        elif whose == 'parents':
            self.parents_score = []
            for parent in self.mating_pool:
                self.parents_score.append(parent.fitness())
            return self.parents_score

    def parent_selection(self, type_ps=None, shift=None, num_parents=None):
        if num_parents is None:
            num_parents = self.pop_size

        if type_ps is None:
            self.mating_pool = self.individuals

        if type_ps == 'sus':
            if shift is None:
                shifted_fitness = self.score
            else:
                shifted_fitness = self.score
                shifted_fitness[:] = [shift - x for x in self.score]
            pre_a = [x / sum(shifted_fitness) for x in shifted_fitness]
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
                        needs_sigma=self.needs_sigma,
                        given_alleles=self.mating_pool[i].gen[:xchg_pt] +
                        self.mating_pool[j].gen[xchg_pt:]))
                self.mating_pool.append(
                    self.Typ(
                        type_gen=self.type_gen,
                        num_alelo=self.num_alelo,
                        needs_sigma=self.needs_sigma,
                        given_alleles=self.mating_pool[j].gen[:xchg_pt] +
                        self.mating_pool[i].gen[xchg_pt:]))
        if tp_xover == 'gbl_dscrt':
            indv_new = np.zeros(self.num_alelo)
            i, j = np.random.choice(range(self.num_alelo), 2)
            for q in range(self.num_alelo):
                if np.random.choice((True, False)):
                    indv_new[q] = self.individuals[i].gen[q]
                else:
                    indv_new[q] = self.individuals[j].gen[q]
            self.mating_pool.append(
                self.Typ(
                    type_gen=self.type_gen,
                    needs_sigma=self.needs_sigma,
                    num_alelo=self.num_alelo,
                    given_alleles=indv_new))
            return indv_new

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

        if tp_mut == 'float':
            for parent in self.mating_pool:
                global_uni = np.random.normal(loc=0.0, scale=1.0)
                local_uni = np.random.normal(loc=0.0, scale=1.0, size=self.num_alelo)

                x_candidato = parent.gen + parent.sigma * local_uni
                indv_candidato = self.Typ(
                    type_gen=self.type_gen,
                    needs_sigma=self.needs_sigma,
                    num_alelo=self.num_alelo,
                    given_alleles=x_candidato)

                indv_candidato.fitness()
                parent.fitness()
                if (indv_candidato.score < parent.score):
                    parent = indv_candidato
                    tau_prime = 1 / (np.sqrt(2 * self.num_alelo))
                    tau = 1 / (np.sqrt(2 * np.sqrt(self.num_alelo)))
                    eps = 10**-10
                    parent.sigma = parent.sigma * np.exp(tau_prime * global_uni + tau * local_uni)
                    for each_sigma in parent.sigma:
                        if each_sigma < eps:
                            each_sigma = eps

    def suvivor_selection(self, tp_selection='ranking'):
        if tp_selection == 'ranking':
            self.fitness(whose='parents')
            idx_sort = list(np.argsort(self.parents_score))
            self.suvivors = [self.mating_pool[i] for i in idx_sort[:self.pop_size]]
            return self.suvivors
