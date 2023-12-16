import torch
import torch.nn as nn
import numpy as np
from z3 import *


def pytorch_to_z3(model):
    """
    Convert a PyTorch Sequential model to a Z3 model.
    Assumes the model alternates between Linear and ReLU layers, ending with Linear.
    """
    # Helper function for ReLU
    def z3_relu(x):
        return z3.If(x > 0, x, 0)

    # Extract weight and bias from PyTorch model
    def extract_weights_and_biases(pytorch_model):
        weights, biases = [], []
        for layer in pytorch_model:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().numpy())
                biases.append(layer.bias.detach().numpy())
            elif isinstance(layer, nn.ReLU):
                pass
            else:
                raise ValueError("Unsupported PyTorch layer type")
        return weights, biases

    weights, biases = extract_weights_and_biases(model)
    num_layers = len(weights)

    # Create Z3 input symbols
    input_size = weights[0].shape[1]
    z3_inputs = [z3.Real(f"nnc_input_{i}__") for i in range(input_size)]
    layer_symbols = [z3_inputs]

    z3_previous_layer = z3_inputs
    for l in range(num_layers):
        w, b = weights[l], biases[l]
        z3_current_layer = []

        for j in range(w.shape[0]):  # for each neuron in this layer
            neuron_symbol = z3.Real(f"nnc_layer_{l}_neuron_{j}__")
            neuron_output = z3.RealVal(b[j]) + sum([z3_previous_layer[i] * z3.RealVal(w[j][i]) for i in range(w.shape[1])])
            
            # Apply ReLU activation for all layers except the last one
            if l != num_layers - 1:
                neuron_output = z3_relu(neuron_output)
            
            z3_current_layer.append(neuron_symbol == neuron_output)
        
        layer_symbols.append(z3_current_layer)
        z3_previous_layer = [z3.Real(f"nnc_layer_{l}_neuron_{j}__") for j in range(len(z3_current_layer))]

    # Create Z3 output symbols for the final layer
    output_size = weights[-1].shape[0]
    z3_outputs_constraints = [i for i in range(output_size)]
    z3_outputs_sym = [z3.Real(f"nnc_output_{i}__") for i in range(output_size)]
    for i in range(output_size):
        z3_outputs_constraints[i] = (z3_outputs_sym[i] == z3_previous_layer[i])
    layer_symbols.append(z3_outputs_constraints)

    return z3_inputs, z3_outputs_sym, layer_symbols


def get_z3_prediction(layer_symbols, z3_inputs, z3_outputs, inputs):
    # Create a Z3 solver
    s = z3.Solver()

    # Bind the input variables to the provided values
    input_binding = [z3_inputs[i] == inputs[i] for i in range(len(z3_inputs))]
    s.add(input_binding)

    # Add the layers to the solver
    for layer in layer_symbols[1:]:
        s.add(layer)

    # Check if there's a solution
    if s.check() == z3.sat:
        m = s.model()
        return np.array([float(m.evaluate(out).as_decimal(10).rstrip('?')) for out in z3_outputs])
    else:
        raise ValueError("No solution found in Z3")


def get_pytorch_prediction_np(model, inputs):
    with torch.no_grad():
        model.eval()
        output = model(torch.Tensor(inputs))
        return output.numpy()


def flatten_model(model):
    """
    Flattens a PyTorch model with nested Sequential modules into a single Sequential module.
    """
    layers = []

    def register_layers(module):
        if isinstance(module, nn.Sequential):
            for sub_module in module.children():
                register_layers(sub_module)
        elif isinstance(module, OnnxablePolicySAC):
            register_layers(module.actor)
        else:
            layers.append(module)

    register_layers(model)

    return nn.Sequential(*layers)


def load_flattened_nn(nn_path):
    onnxable_model = torch.load(nn_path)
    onnxable_model = flatten_model(onnxable_model)
    return onnxable_model


class OnnxablePolicySAC(torch.nn.Module):
    def __init__(self, sb3_SAC_model):
        super(OnnxablePolicySAC, self).__init__()
        self.actor = torch.nn.Sequential(
            sb3_SAC_model.policy.actor.latent_pi, sb3_SAC_model.policy.actor.mu
        )

    def forward(self, observation):
        return self.actor(observation)