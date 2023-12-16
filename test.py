from sys import argv
from pytorch_fc_to_smt import *

from z3 import *
import torch
import torch.nn as nn
from pytorch_fc_to_smt import *
import numpy as np
from tqdm import tqdm

def is_state_in_inv(state):
    if (0.25 <= state[0] <= 0.95
        and 
        0.55 <= state[1] <= 0.95):
        return True
    else:
        return False

if __name__ == "__main__":
    if argv[1] == "pt_to_z3":
        VERBOSE_SMT_FORMULA = False
        nn_models_dir = "assets/non-det-maze/nn-models"
        net_arch_list = [
            [32, 32], 
            [40, 40],
            [48, 48], 
            [56, 56],
            [64, 64],
            [128, 128],
        ]
        for net_arch in net_arch_list:
            print("\n--------------------------")
            print(f"Testing PyTorch=Z3 with NN arch {net_arch}...")
            # NN names
            nn_id = "".join([f"_{i}" for i in net_arch])
            nn_path = f"{nn_models_dir}/maze_2d_{nn_id}"

            # Load Onnxable model
            onnxable_model = torch.load(nn_path, map_location=torch.device("cpu")).cpu()
            onnxable_model = flatten_model(onnxable_model)



            z3_in_sym, z3_out_sym, z3_layer_sym = pytorch_to_z3(onnxable_model)

            if VERBOSE_SMT_FORMULA:
                from torchinfo import summary
                summary(onnxable_model)
                print(onnxable_model)


            if VERBOSE_SMT_FORMULA:
                print(f"z3_in_sym: {z3_in_sym}")
                print(f"z3_out_sym: {[i.sexpr() for i in z3_out_sym]}")
                print("--------------------------")
                print(f"layer num of NN: {len(z3_layer_sym)}")
                print("--------------------------")
                for i, layer in enumerate(z3_layer_sym):
                    print("+++++++++++++++++++++++++++++++")
                    print(f"|\tlayer {i}: sym num {len(layer)}")

                    for j, sym in enumerate(layer):
                        print("===============================")
                        print(f"|\t\tsym {j}: {sym.sexpr()}")

            # Test
            num_samples = 1000
            comparison_tolerance = 1e-5
            print(f"Testing {num_samples} samples (tolerance={comparison_tolerance})...")
            inputs = np.random.randn(num_samples, len(z3_in_sym))
            pytorch_outputs = [
                get_pytorch_prediction_np(onnxable_model, sample) for sample in tqdm(inputs, desc="PyTorch Predictions")
            ]
            z3_outputs_values = [
                get_z3_prediction(z3_layer_sym, z3_in_sym, z3_out_sym, sample)
                for sample in tqdm(inputs, desc="Z3 Predictions")
            ]

            # Compare
            for i in tqdm(range(num_samples), desc="Comparing Outputs..."):
                if (abs(z3_outputs_values[i] - pytorch_outputs[i]) > comparison_tolerance).any():
                    print("Discrepancy detected!")
                    print("PyTorch output:", pytorch_outputs[i])
                    print("Z3 output:", z3_outputs_values[i])

            # Note: Due to the inherent floating-point imprecisions, we might not get exactly the same results, but they should be very close.


    elif argv[1] == "indeed_counterexample":
        print("\n--------------------------")
        print(f"Testing Falsifying Predicate ...")
        nn_models_dir = "assets/det-maze/nn-models"
        # NN names
        nn_id = ''.join([f'_{i}' for i in [128, 128]])
        nn_path = f'{nn_models_dir}/maze_2d_{nn_id}'

        loaded_nn = torch.load(nn_path).cuda()

        # this is the falsifying state predicate
        # And(And(x_0 >= 601/640, x_0 <= 19/20),
        #     And(x_1 >= 57/80, x_1 <= 23/32))
        
        x = np.random.uniform(low=601/640, high=19/20, size=(10000, 1))
        y = np.random.uniform(low=57/80, high=23/32, size=(10000, 1))
        input = torch.Tensor(np.concatenate((x, y), axis=1)).cuda()
        print(f"Samples: {input.shape[0]}")
        output = loaded_nn(input)
        next_state = input + 0.1 * output
        # all next state out of ind inv
        next_state_np = [x.detach().cpu().numpy() for x in next_state]
        
        for i in tqdm(next_state_np, desc="Checking Indeed Counter-examples"):
            if is_state_in_inv(i):
                print(f"Next state in inv: {i}")

    else:
        raise ValueError("Invalid argument")