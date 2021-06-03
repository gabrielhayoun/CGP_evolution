import cgp
from minatar import Environment

NUM_FRAMES = 1000
MAX_EVALS = 5000
GAME="breakout"

def argmax(policy):
    i, max = 0, policy[0]
    for j in range(len(policy)):
        if policy[j] > max:
            max = policy[j]
            i = j
    return i

def convertState (state, nbObjects):
    newState = [] 
    for i in range (len(state)):
        for j in range (len(state[0])):
            c=nbObjects
            while c>0 and not(state[i][j][c-1]) :
                c-=1
            pixelValue = c/nbObjects
            newState.append(pixelValue)
    return newState



def play(individual, display=False):
    env = Environment(GAME, sticky_action_prob=0.0, random_seed=0)
    env.reset()
    nbObjects = len(env.state()[0][0])
    is_terminated = False
    total_reward = 0.0
    t = 0
    # print(individual.idx)
    while (not is_terminated) and t < NUM_FRAMES:
        eval = individual.to_func()
        # print("t=",t)
        # print(convertState(env.state(),nbObjects))
        policy = eval(*convertState(env.state(),nbObjects))
        # print("policy=",policy)
        action = argmax(policy)*2+1
        # print("action =",action)
        reward, is_terminated = env.act(action)
        total_reward += reward
        t += 1
        if display and individual.idx%10 == 0:
            env.display_state(1)
    if display and individual.idx%10 == 0:
        env.close_display()
    return total_reward


# -------------Objective Function-------------
def objective(individual):
    individual.fitness = play(individual)
    return individual


# -----------Parameters of population, genome, evolutionary algorithm (EA) and evolve function ------------------
population_params = {"n_parents": 10,  "seed": 8188211}

genome_params = {
    "n_inputs": 100,
    "n_outputs": 2,
    "n_columns": 100,
    "n_rows": 2,
    "levels_back": 100,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Pow, cgp.ConstantFloat),
}

ea_params = {"n_offsprings": 10, "tournament_size": 3, "mutation_rate": 0.1,"n_processes":6}

evolve_params = {"max_generations": 1000}

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
