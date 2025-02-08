# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import numpy as np

# Load Network Data 
def load_network(file_path):
    """Loads the network file and checks for required columns"""
    df = pd.read_csv(file_path, delimiter=";", compression="gzip")

    expected_columns = {"link", "from_node", "to_node", "length", "freespeed", "capacity", "lanes", "modes"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("Network file missing required columns!")

    return df

# Load Travel Data
def load_travel_data(file_path):
    """Loads travel data from previous simulation results"""
    df = pd.read_csv(file_path, delimiter=";", compression="gzip")

    expected_columns = {"person", "start_link", "end_link", "trav_time", "distance"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("Travel file missing required columns!")

    return df

# Load Population XML 
def load_population(file_path):
    """Loads agents from XML population file"""
    tree = ET.parse(file_path)
    root = tree.getroot()

    agents = []
    for person in root.findall("person"):
        person_id = person.get("id")
        plan = person.find("plan")
        activities = plan.findall("act")
        legs = plan.findall("leg")

        if len(activities) >= 2 and len(legs) >= 1:
            agents.append((person_id, activities[0], legs[0], activities[1]))  # (ID, start act, leg, end act)

    return agents, tree

def convert_time_to_seconds(time_str):
    """Convert hh:mm:ss format to seconds"""
    time_parts = time_str.split(":")
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 2:  # Handle "mm:ss" case
        hours, minutes, seconds = 0, *map(int, time_parts)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    return hours * 3600 + minutes * 60 + seconds

# Fitness Function
def fitness(individual, agents, travel_data):
    """Evaluates the fitness by minimizing travel time and distance"""
    total_travel_time = 0
    total_distance = 0

    for i, dep_time in enumerate(individual):
        person_id, act, leg, _ = agents[i]

        # Get actual travel time and distance from previous simulation
        trip_info = travel_data[travel_data["person"] == person_id]
        if not trip_info.empty:
            actual_travel_time = convert_time_to_seconds(trip_info.iloc[0]["trav_time"])
            actual_distance = float(trip_info.iloc[0]["distance"])  # Convert to float
        else:
            actual_travel_time = 3600  # Default 1 hour
            actual_distance = 5000  # Default 5 km

        # Convert to seconds
        time_parts = dep_time.split(":")
        hours, minutes, seconds = map(int, time_parts) if len(time_parts) == 3 else (int(time_parts[0]), int(time_parts[1]), 0)
        departure_seconds = (hours * 3600) + (minutes * 60) + seconds

        # Update XML departure time
        act.set("end_time", dep_time)

        total_travel_time += actual_travel_time
        total_distance += actual_distance

    return (1 / total_travel_time, 1 / total_distance)  # Minimize both

# GA Setup 
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Two objectives: time & distance
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_dep_time", lambda: f"{random.randint(6, 9):02}:{random.randint(0, 59):02}:{random.randint(0, 59):02}")  # Random time between 6:00 - 9:00 AM
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_dep_time, n=1000)  # Assuming 1000 agents
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutate(individual):
    """Mutates departure time slightly"""
    if random.random() < 0.5:
        h, m, s = map(int, individual[random.randint(0, len(individual) - 1)].split(":"))
        h = min(max(h + random.choice([-1, 1]), 6), 9)  # Keep between 6-9 AM
        m = min(max(m + random.choice([-15, 15]), 0), 59)
        individual[random.randint(0, len(individual) - 1)] = f"{h:02}:{m:02}:{s:02}"
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

#Run the Genetic Algorithm
def run_ga(population_file, travel_file, network_file, output_file="optimized_population.xml"):
    """Runs the genetic algorithm using real travel times and optimizes both time & route"""
    agents, tree = load_population(population_file)
    travel_data = load_travel_data(travel_file)
    network_data = load_network(network_file)

    toolbox.register("evaluate", fitness, agents=agents, travel_data=travel_data)

    # Create population
    pop = toolbox.population(n=100)

    # Run GA
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

    # Update XML file with optimized departure times
    for i, dep_time in enumerate(pop[0]):  # Best individual
        _, act, _, _ = agents[i]
        act.set("end_time", dep_time)

    tree.write(output_file)
    print(f"Optimized population saved as {output_file}")

# Run the GA
if __name__ == "__main__":
    run_ga("generated_population_2.xml", "output_legs.csv_initial.gz", "output_links.csv.gz")
