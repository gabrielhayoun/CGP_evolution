import cgp
from minatar import Environment
from Custom_functions import *

NUM_FRAMES = 1000
MAX_EVALS = 5000
GAME="breakout"

max_fitness=0

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
        if display:
            env.display_state(1)
    if display:
        env.close_display()
    return total_reward


# -------------Objective Function-------------
def objective(individual):
    individual.fitness = play(individual,0)
    return individual


# -----------Parameters of population, genome, evolutionary algorithm (EA) and evolve function ------------------
population_params = {"n_parents": 10,  "seed": 8188212}

genome_params = {
    "n_inputs": 100,
    "n_outputs": 3,
    "n_columns": 20,
    "n_rows": 10,
    "levels_back": 10,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, Div, 
                    #Pow, Pow2, Sqrt, Double, Cos, Sin, Inv, Tanh, Gaussian, 
                    Min, Max, Transistor, Compare, cgp.ConstantFloat),
}

ea_params = {"n_offsprings": 10, "tournament_size": 2, "mutation_rate": 0.6,"n_processes":10}

evolve_params = {"max_generations": 10000}#int(MAX_EVALS/population_params["n_parents"])}

# --------Initialize a population and an EA instance
pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)

# ---------Record information------------
history = {}
history["fitness_parents"] = []


def colorFunctions (str):
    funcList = ["Add", "Sub", "Mul", "Div", "Pow", "Pow2", "Sqrt", "Double", "Cos", "Sin", "Inv", "Tanh", "Gaussian", "Min", "Max", "Transistor", "Compare", "ConstantFloat"]
    for f in funcList :
        index=str.find(f)
        if index !=-1:
            str=str[0:index]+"\033[1;31m"+str[index:index+len(f)]+"\033[1;00m"+str[index+len(f):]
            while True :
                index=str.find(f,index+len(f)+10)
                if index !=-1:
                    str=str[0:index]+"\033[1;31m"+str[index:index+len(f)]+"\033[1;00m"+str[index+len(f):]
                else :
                    break
    return str

def recording_callback(pop):
    # history["fitness_parents"].append(pop.fitness_parents())
    if pop.generation%50 == 0 :
        play(pop.champion,True)
        print("\n"+colorFunctions(cgp.CartesianGraph(pop.champion.genome).print_active_nodes())+"\n")


# ----------Execute the evolution------------
cgp.evolve(pop, objective, ea, **evolve_params, print_progress=True, callback=recording_callback)
