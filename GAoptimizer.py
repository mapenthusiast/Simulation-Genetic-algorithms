# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pandas as pd
import random
from deap import base, creator, tools, algorithms

#  Load Departure Times from Simulation Results(output_legs.csv.gz)

def get_departure_times(file_path):
    df = pd.read_csv(file_path, compression="gzip", delimiter=";", encoding="utf-8")
    
    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Ensure relevant columns exist
    required_columns = ["person", "dep_time"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Required columns not found in the simulation output! Missing: {missing}")

    dep_times = df.groupby("person")["dep_time"].first().to_dict()  # First departure per agent
    return dep_times



# Load agents from Population XML

def load_population(file_path):
   
    tree = ET.parse(file_path)
    root = tree.getroot()

    agents = []
    for person in root.findall("person"):
        person_id = person.get("id")
        plan = person.find("plan")
        activities = plan.findall("act")
        legs = plan.findall("leg")

        if len(activities) >= 2:
            agents.append((person_id, activities[0], legs[0], activities[1]))  # (ID, start act, leg, end act)

    return agents, tree


# Fitness Function (Uses Departure Times from previous simulation)

def fitness(individual, agents, dep_times):
    
    total_travel_time = 0

    for i, dep_time in enumerate(individual):
        person_id, act, _, _ = agents[i]

        # Get actual departure time from the simulation results
        actual_dep_time = dep_times.get(person_id, "07:00:00")  # Default to 7 AM if missing

        # Update end_time based on the individual's suggested departure
        act.set("end_time", actual_dep_time)

        # Convert to seconds
        time_parts = actual_dep_time.split(":")
        hours, minutes, seconds = map(int, time_parts) if len(time_parts) == 3 else (int(time_parts[0]), int(time_parts[1]), 0)

        total_travel_time += (hours * 3600) + (minutes * 60) + seconds

    return (1 / total_travel_time,)  # Minimize travel time


#  Genetic Algorithm Setup
# ===========================

# Create the Fitness and Individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Register attributes for generating individuals
toolbox.register("attr_dep_time", lambda: random.randint(21600, 32400))  # 6:00 AM - 9:00 AM in seconds
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_dep_time, n=1000)  # Assuming 1000 agents
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Mutation: Small random changes
def mutate(individual):
    """Mutate the departure time slightly"""
    if random.random() < 0.5:
        individual[random.randint(0, len(individual) - 1)] += random.randint(-300, 300)  # Adjust by ±5 min
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


# Run the Genetic Algorithm


def run_ga(population_file, results_file, output_file="optimized_population.xml"):
    """
    Runs the genetic algorithm using real departure times from the last simulation.
    """
    # Load agents from XML
    agents, tree = load_population(population_file)

    # Extract real departure times
    dep_times = get_departure_times(results_file)

    # Define GA settings
    toolbox.register("evaluate", fitness, agents=agents, dep_times=dep_times)

    # Create population
    pop = toolbox.population(n=100)

    # Run GA
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

    # Update the population XML with optimized departure times
    for i, dep_time in enumerate(pop[0]):  # Take the best individual
        _, act, _, _ = agents[i]
        hours = dep_time // 3600
        minutes = (dep_time % 3600) // 60
        seconds = dep_time % 60
        act.set("end_time", f"{hours:02}:{minutes:02}:{seconds:02}")

    # Save new XML file
    tree.write(output_file)
    print(f"Optimized population saved as {output_file}")


#  Run the Optimization
if __name__ == "__main__":
    run_ga("generated_population_2.xml", "output_legs.csv_optimized.gz")
