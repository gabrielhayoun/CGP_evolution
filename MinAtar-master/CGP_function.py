import cgp
from minatar import GUI

# -------------Objective Function-------------
def objective(individual):
    individual.fitness = ...
    return individual

# -----------Parameters of population, genome, evolutionary algorithm (EA) and evolve function ------------------
population_params = {"n_parents": 10, "mutation_rate": 0.5, "seed": 8188211}

genome_params = {
     "n_inputs": 100,
     "n_outputs": 3,
     "n_columns": 10,
     "n_rows": 1,
     "levels_back": 10,
     "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Div, cgp.ConstantFloat),
     }

ea_params = {"n_offsprings": 10, "tournament_size": 2, "n_processes": 2}

evolve_params = {"max_generations": 1000, "min_fitness": 0.0}

# --------Initialize a population and an EA instance
pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)

# ---------Record information------------
history = {}
history["fitness_parents"] = []
def recording_callback(pop):
    history["fitness_parents"].append(pop.fitness_parents())


# ----------Execute the evolution------------
cgp.evolve(pop, objective, ea, **evolve_params, print_progress=True, callback=recording_callback)