import math
import numpy as np

np.random.seed(99)
def quantitySkew(numberSample, numberClient):
    minSample = math.floor(0.1 / numberClient * numberSample) 
    remaining_elements = numberSample - numberClient * minSample
    group_sizes = [minSample] * numberClient
    
    # Define the integers and their corresponding probabilities
    integers = [0, 1, 2, 3, 4]
    probabilities = [0.05, 0.1, 0.5, 0.15, 0.2]  # Make sure probabilities sum up to 1
    
    for _ in range(remaining_elements):
        group_sizes[np.random.choice(integers, p=probabilities)] += 1
    return group_sizes
    