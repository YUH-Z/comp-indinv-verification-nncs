from sys import argv
import torch

from pytorch_fc_to_smt import *
from sts import *

import z3


if __name__ == '__main__':
    net_arch_list = [
        [32, 32], 
        [40, 40],
        [48, 48], 
        [56, 56],
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512],
        [1024, 1024],
    ]
    if argv[1] == 'det':
        nn_models_dir = 'assets/det-maze/nn-models'
        smt_output_dir = 'assets/det-maze/mono-smt-queries'
        for net_arch in net_arch_list:
            # NN names
            nn_id = ''.join([f'_{i}' for i in net_arch])
            nn_path = f'{nn_models_dir}/maze_2d_{nn_id}'

            loaded_nn = torch.load(nn_path).cpu()


            z3_in_sym, z3_out_sym, z3_layer_sym = pytorch_to_z3(loaded_nn)

            nnsts = STS()

            # We already have the input and output symbols for the NN in previous cells
            # So we only need to compose them with the environment

            # Since the input to the NN controller is just the state of the system,
            # we actually don't need to define any new variables
            # the variable definitions here is just for convenience
            x, x_next = nnsts.make_variable("x", Real)
            y, y_next = nnsts.make_variable("y", Real)

            # Initial state ($0.3 \le x \le 0.4 \land 0.6 \le y \le 0.7$)
            @nnsts.make_initial_condition
            def initial():
                return Z3I.cap(x <= 0.4, x >= 0.3, y <= 0.7, y >= 0.6)


            # Encode (x,y == NN_input) and
            # neural network constraints (z3_layer_sym[1:],
            # because the first layer is input symbols) here
            @nnsts.make_transition_relation
            def transition():
                return Z3I.cap(
                    x == Z3I(z3_in_sym[0]),
                    y == Z3I(z3_in_sym[1]),
                    x_next == x + 0.1 * Z3I(z3_out_sym[0]),
                    y_next == y + 0.1 * Z3I(z3_out_sym[1]),
                    *[
                        Z3I(neuron_constraint)
                        for layer_constraints in z3_layer_sym[1:]
                        for neuron_constraint in layer_constraints
                    ]
                )


            # - We want to prove that the system will never reach the red region
            #   - $x < 0.25 \lor$
            #   - $x > 0.95 \lor$
            #   - $y < 0.55 \lor$
            #   - $y > 0.95$

            @nnsts.make_invariant
            def invariant():
                return Z3I.cap(
                    x <= 0.95,
                    x >= 0.25,
                    y <= 0.95,
                    y >= 0.55
                )
            
            initial()
            transition()
            invariant()

            print(f"+++=============================+++")
            print(f"Gen SMT Form for Net Arch: {net_arch}")
            print(f"+++=============================+++")
            ind_form = nnsts.invariant_inductiveness.z3_var
            s = z3.Solver()
            s.add(z3.Not(ind_form))
            ind_smt2 = s.to_smt2()

            z3_timeout = "(set-option :timeout 3600000)"
            z3_random_seed = "(set-option :random-seed 42)"
            ind_smt2 = "\n".join([z3_timeout, z3_random_seed, ind_smt2])

            with open(f"{smt_output_dir}/ind_smt2_{nn_id}.smt2", "w") as f:
                f.write(ind_smt2)
            
            print()
        
    if argv[1] == 'non-det':
        nn_models_dir = 'assets/non-det-maze/nn-models'
        smt_output_dir = 'assets/non-det-maze/mono-smt-queries'
        for net_arch in net_arch_list:
            # NN names
            nn_id = ''.join([f'_{i}' for i in net_arch])
            nn_path = f'{nn_models_dir}/maze_2d_{nn_id}'

            loaded_nn = torch.load(nn_path, map_location=torch.device('cpu')).cpu()


            z3_in_sym, z3_out_sym, z3_layer_sym = pytorch_to_z3(loaded_nn)

            nnsts = STS()

            # We already have the input and output symbols for the NN in previous cells
            # So we only need to compose them with the environment

            # Since the input to the NN controller is just the state of the system,
            # we actually don't need to define any new variables
            # the variable definitions here is just for convenience
            x, x_next = nnsts.make_variable("x", Real)
            y, y_next = nnsts.make_variable("y", Real)

            # Initial state ($0.3 \le x \le 0.4 \land 0.6 \le y \le 0.7$)
            @nnsts.make_initial_condition
            def initial():
                return Z3I.cap(x <= 0.4, x >= 0.3, y <= 0.7, y >= 0.6)


            # Encode (x,y == NN_input) and
            # neural network constraints (z3_layer_sym[1:],
            # because the first layer is input symbols) here
            c = Z3I(z3.Real('c'))
            @nnsts.make_transition_relation
            def transition():
                return Z3I.cap(
                    x == Z3I(z3_in_sym[0]),
                    y == Z3I(z3_in_sym[1]),
                    Z3I.cap(
                        x_next == x + 0.1 * c * Z3I(z3_out_sym[0]),
                        y_next == y + 0.1 * c * Z3I(z3_out_sym[1]),
                        c <= 1.0,
                        c >= 0.5,
                    ),
                    *[
                        Z3I(neuron_constraint)
                        for layer_constraints in z3_layer_sym[1:]
                        for neuron_constraint in layer_constraints
                    ]
                )


            # - We want to prove that the system will never reach the red region
            #   - $x < 0.25 \lor$
            #   - $x > 0.95 \lor$
            #   - $y < 0.55 \lor$
            #   - $y > 0.95$

            @nnsts.make_invariant
            def invariant():
                return Z3I.cap(
                    x <= 0.95,
                    x >= 0.25,
                    y <= 0.95,
                    y >= 0.55
                )
            
            initial()
            transition()
            invariant()

            print(f"+++=============================+++")
            print(f"Gen SMT Form for Net Arch: {net_arch}")
            print(f"+++=============================+++")
            ind_form = nnsts.invariant_inductiveness.z3_var
            s = z3.Solver()
            s.add(z3.Not(ind_form))
            ind_smt2 = s.to_smt2()

            z3_timeout = "(set-option :timeout 3600000)"
            z3_random_seed = "(set-option :random-seed 42)"
            ind_smt2 = "\n".join([z3_timeout, z3_random_seed, ind_smt2])

            with open(f"{smt_output_dir}/ind_smt2_{nn_id}.smt2", "w") as f:
                f.write(ind_smt2)
            
            print()