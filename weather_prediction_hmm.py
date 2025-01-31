import numpy as np
import random

# Hidden Markov Model (HMM) for Weather Prediction

# Define states and observations
states = ['Sunny', 'Rainy', 'Cloudy']
observations = ['Dry', 'Damp', 'Wet']

# Transition probabilities (A)
transition_prob = {
    'Sunny': {'Sunny': 0.7, 'Rainy': 0.2, 'Cloudy': 0.1},
    'Rainy': {'Sunny': 0.3, 'Rainy': 0.5, 'Cloudy': 0.2},
    'Cloudy': {'Sunny': 0.4, 'Rainy': 0.3, 'Cloudy': 0.3}
}

# Emission probabilities (B)
emission_prob = {
    'Sunny': {'Dry': 0.6, 'Damp': 0.3, 'Wet': 0.1},
    'Rainy': {'Dry': 0.1, 'Damp': 0.4, 'Wet': 0.5},
    'Cloudy': {'Dry': 0.3, 'Damp': 0.5, 'Wet': 0.2}
}

# Initial state probabilities (π)
initial_prob = {'Sunny': 0.5, 'Rainy': 0.3, 'Cloudy': 0.2}

# Viterbi Algorithm for most likely weather sequence
def viterbi(obs_seq, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = start_p[state] * emit_p[state][obs_seq[0]]
        path[state] = [state]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs_seq)):
        V.append({})
        new_path = {}
        
        for curr_state in states:
            (prob, state) = max((V[t-1][prev_state] * trans_p[prev_state][curr_state] * emit_p[curr_state][obs_seq[t]], prev_state) for prev_state in states)
            V[t][curr_state] = prob
            new_path[curr_state] = path[state] + [curr_state]
        
        path = new_path
    
    # Find the final most probable state
    (prob, state) = max((V[-1][s], s) for s in states)
    return path[state], prob

# Example observation sequence
obs_sequence = ['Dry', 'Damp', 'Wet', 'Damp', 'Dry']

# Predict the most likely weather sequence
most_likely_states, probability = viterbi(obs_sequence, states, initial_prob, transition_prob, emission_prob)

print(f"Most Likely Weather Sequence: {most_likely_states}")
print(f"Probability of this sequence: {probability:.5f}")

# Generate a random weather sequence based on transition probabilities
def generate_weather_sequence(length, start_state):
    sequence = [start_state]
    for _ in range(length - 1):
        next_state = random.choices(list(transition_prob[start_state].keys()), 
                                    weights=transition_prob[start_state].values())[0]
        sequence.append(next_state)
        start_state = next_state
    return sequence

# Generate a random observation sequence based on emission probabilities
def generate_observation_sequence(weather_seq):
    obs_seq = []
    for state in weather_seq:
        obs = random.choices(list(emission_prob[state].keys()), 
                             weights=emission_prob[state].values())[0]
        obs_seq.append(obs)
    return obs_seq

# Generate and print a random weather sequence and its observations
random_weather_seq = generate_weather_sequence(10, 'Sunny')
random_obs_seq = generate_observation_sequence(random_weather_seq)
print(f"Generated Weather Sequence: {random_weather_seq}")
print(f"Generated Observation Sequence: {random_obs_seq}")
