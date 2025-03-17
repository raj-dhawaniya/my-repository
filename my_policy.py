import numpy as np

def policy_action(params, observation):
    W = params[:32].reshape(8, 4)  # 8 inputs x 4 outputs
    b = params[32:].reshape(4)     # 4 biases
    logits = np.dot(observation, W) + b
    return np.argmax(logits)