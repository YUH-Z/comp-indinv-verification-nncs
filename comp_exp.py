import time
import torch
import pathlib
from collections import deque

from pytorch_fc_to_smt import *
from autoLIRPA_utils import *

from itertools import product

from auto_LiRPA import BoundedModule

import z3

from argparse import ArgumentParser

def is_state_in_inv(state):
    if (0.25 <= state[0] <= 0.95
        and 
        0.55 <= state[1] <= 0.95):
        return True
    else:
        return False


def split_bound(bound):
    # split the bound into two sub-bounds
    bound_split = bound 
    sub_bound1 = torch.Tensor([bound_split[0], (bound_split[0] + bound_split[1]) / 2])
    sub_bound2 = torch.Tensor([(bound_split[0] + bound_split[1]) / 2, bound_split[1]])

    # calculate the centroids of the sub-bounds
    sub_bound1_c = (sub_bound1[0] + sub_bound1[1]) / 2
    sub_bound2_c = (sub_bound2[0] + sub_bound2[1]) / 2

    return (sub_bound1, sub_bound1_c), (sub_bound2, sub_bound2_c)


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
    parser = ArgumentParser("2D Maze Experiments")

    parser.add_argument('--nn-dir', '--nn_models_dir', type=str, help='Path to the directory containing the NN models')

    parser.add_argument('-e', '--experiment_name', choices=['non-det', 'det'], help='Experiment name (non-det or det)')

    parser.add_argument('-ms', '--max_split', type=int, default=1000, help='Maximum split')

    parser.add_argument('-d', '--debug', default=False, action="store_true", help='Debug mode')

    args = parser.parse_args()

    nn_models_dir = args.nn_dir
    assert pathlib.Path(nn_models_dir).is_dir(), f"{nn_models_dir} is not a directory"

    experiment_name = args.experiment_name
    MAX_SPLIT = args.max_split
    DEBUG = args.debug

    ind_inv = torch.Tensor([[0.25, 0.55], [0.95, 0.95]])
    for net_arch in net_arch_list:
        # NN names
        nn_id = ''.join([f'_{i}' for i in net_arch])
        nn_path = f'{nn_models_dir}/maze_2d_{nn_id}'

        loaded_nn = torch.load(nn_path)

        bounded_model = BoundedModule(loaded_nn, torch.empty((1, 2)))

        start_time = time.time()

        queue_to_verify = deque()
        queue_to_verify.append(
            (torch.Tensor([[0.25, 0.55], [0.95, 0.95]]),
             torch.Tensor([[0.6, 0.75]]))) # Centroid of the region, for computing bounds

        num_split = 0
        num_smt_queries = 0
        num_nnv_queries = 0
        num_clauses = 0
        bridge_list = []

        if experiment_name == 'det':
            print(f"\nVerifying Det 2D maze with [{', '.join([str(i) for i in net_arch])}] NN Controller")
            while num_split < MAX_SPLIT:
                if not (queue_to_verify):
                    print("Verified!")
                    break
                
                curr_bound, curr_c = queue_to_verify.popleft()

                pred_bound = compute_bounds(
                    bounded_model,
                    input_l=curr_bound[0].unsqueeze(0),
                    input_u=curr_bound[1].unsqueeze(0),
                    x=curr_c
                )
                num_nnv_queries += 1
                trans_bound_l = curr_bound[0].unsqueeze(0) + 0.1 * pred_bound[0].cpu()
                trans_bound_u = curr_bound[1].unsqueeze(0) + 0.1 * pred_bound[1].cpu()


                x_l = curr_bound[0].unsqueeze(0).cpu().numpy()[0]
                x_u = curr_bound[1].unsqueeze(0).cpu().numpy()[0]

                curr_bound_z3 = []
                x_vars = []
                for i in range(len(x_l)):
                    z3_var = z3.Real(f'x_{i}')
                    x_vars.append(z3_var)
                    curr_bound_z3.append(z3.And(
                        z3_var >= x_l[i],
                        z3_var <= x_u[i]
                    ))

                
                a_l = pred_bound[0].cpu().numpy()[0]
                a_u = pred_bound[1].cpu().numpy()[0]

                # make z3 variables in these bounds
                pred_bounds_z3 = []
                a_vars = []
                for i in range(len(a_l)):
                    z3_var = z3.Real(f'a_{i}')
                    a_vars.append(z3_var)
                    pred_bounds_z3.append(z3.And(
                        z3_var >= a_l[i],
                        z3_var <= a_u[i]
                    ))
                
                next_var_z3 = []
                trans_cond_z3 = []
                for i in range(len(a_vars)):
                    z3_var = z3.Real(f'x_next_{i}')
                    next_var_z3.append(z3_var)
                    trans_cond_z3.append(
                        x_vars[i] + 0.1 * a_vars[i] == z3_var
                    )


                left_imp_z3 = z3.And(
                    z3.And(*curr_bound_z3),
                    z3.And(*pred_bounds_z3),
                    z3.And(*trans_cond_z3)
                )

                ind_inv_cpu = ind_inv.cpu().numpy()
                inv_next_z3 = z3.And(
                    next_var_z3[0] >= ind_inv_cpu[0][0],
                    next_var_z3[0] <= ind_inv_cpu[1][0],
                    next_var_z3[1] >= ind_inv_cpu[0][1],
                    next_var_z3[1] <= ind_inv_cpu[1][1]
                )

                ind_check = z3.Implies(left_imp_z3, inv_next_z3)
                fal_check = z3.Implies(left_imp_z3, z3.Not(inv_next_z3))

                s_ind = z3.Solver()
                s_ind.add(z3.Not(ind_check))
                s_fal = z3.Solver()
                s_fal.add(z3.Not(fal_check))

                if s_ind.check() == z3.unsat:
                    num_smt_queries += 1
                    num_clauses += 1
                    bridge_list.append(z3.And(z3.And(*curr_bound_z3), z3.And(*pred_bounds_z3)))
                    if DEBUG:
                        print(f'ind check passed: {ind_check}')
                elif s_fal.check() == z3.unsat:
                    print(f"falsifying predicate found! \n{z3.And(*curr_bound_z3)}")
                    num_smt_queries += 2
                    if DEBUG:
                        print(f'fals check: {fal_check}')
                    break
                else:
                    num_smt_queries += 2
                    # split the bounds
                    if DEBUG:
                        print(f'splitting {curr_bound.cpu().detach().numpy()}')
                    tcb = curr_bound.transpose(0, 1)
                    sub_bound_x1, sub_bound_x2 = split_bound(tcb[0])
                    sub_bound_y1, sub_bound_y2 = split_bound(tcb[1])

                    for x, y in product([sub_bound_x1, sub_bound_x2], [sub_bound_y1, sub_bound_y2]):
                        lb = torch.Tensor([x[0][0], y[0][0]])
                        ub = torch.Tensor([x[0][1], y[0][1]])
                        c = torch.Tensor([x[1], y[1]]).unsqueeze(0)
                        bound = torch.cat([lb.unsqueeze(0), ub.unsqueeze(0)], dim=0)

                        # print(x, y)
                        # print(lb, ub, c, bound)
                        queue_to_verify.append((bound, c))
                    
                    num_split += 1
                
            if num_split >= MAX_SPLIT:
                print("!MAX SPLIT REACHED!")



        elif experiment_name == "non-det":
            print(f"\nVerifying NDet 2D maze with [{', '.join([str(i) for i in net_arch])}] NN Controller")
            while num_split < MAX_SPLIT:
                if not (queue_to_verify):
                    print("Verified!")
                    break
                
                curr_bound, curr_c = queue_to_verify.popleft()

                pred_bound = compute_bounds(
                    bounded_model,
                    input_l=curr_bound[0].unsqueeze(0),
                    input_u=curr_bound[1].unsqueeze(0),
                    x=curr_c
                )
                num_nnv_queries += 1
                trans_bound_l = curr_bound[0].unsqueeze(0) + 0.1 * pred_bound[0].cpu()
                trans_bound_u = curr_bound[1].unsqueeze(0) + 0.1 * pred_bound[1].cpu()

                x_l = curr_bound[0].unsqueeze(0).cpu().numpy()[0]
                x_u = curr_bound[1].unsqueeze(0).cpu().numpy()[0]

                curr_bound_z3 = []
                x_vars = []
                for i in range(len(x_l)):
                    z3_var = z3.Real(f'x_{i}')
                    x_vars.append(z3_var)
                    curr_bound_z3.append(z3.And(
                        z3_var >= x_l[i],
                        z3_var <= x_u[i]
                    ))

                
                a_l = pred_bound[0].cpu().numpy()[0]
                a_u = pred_bound[1].cpu().numpy()[0]

                # make z3 variables in these bounds
                pred_bounds_z3 = []
                a_vars = []
                for i in range(len(a_l)):
                    z3_var = z3.Real(f'a_{i}')
                    a_vars.append(z3_var)
                    pred_bounds_z3.append(z3.And(
                        z3_var >= a_l[i],
                        z3_var <= a_u[i]
                    ))
                
                c_var = z3.Real(f'c')
                next_var_z3 = []
                trans_cond_z3 = []
                for i in range(len(a_vars)):
                    z3_var = z3.Real(f'x_next_{i}')
                    next_var_z3.append(z3_var)
                    trans_cond_z3.append(
                        z3.And(
                            c_var <= 1.0,
                            c_var >= 0.5,
                            x_vars[i] + 0.1 * c_var * a_vars[i] == z3_var,
                        )
                    )


                left_imp_z3 = z3.And(
                    z3.And(*curr_bound_z3),
                    z3.And(*pred_bounds_z3),
                    z3.And(*trans_cond_z3)
                )

                ind_inv_cpu = ind_inv.cpu().numpy()
                inv_next_z3 = z3.And(
                    next_var_z3[0] >= ind_inv_cpu[0][0],
                    next_var_z3[0] <= ind_inv_cpu[1][0],
                    next_var_z3[1] >= ind_inv_cpu[0][1],
                    next_var_z3[1] <= ind_inv_cpu[1][1]
                )

                ind_check = z3.Implies(left_imp_z3, inv_next_z3)
                fal_check = z3.Implies(left_imp_z3, z3.Not(inv_next_z3))

                s_ind = z3.Solver()
                s_ind.add(z3.Not(ind_check))
                s_fal = z3.Solver()
                s_fal.add(z3.Not(fal_check))

                if s_ind.check() == z3.unsat:
                    num_smt_queries += 1
                    num_clauses += 1
                    bridge_list.append(z3.And(z3.And(*curr_bound_z3), z3.And(*pred_bounds_z3)))
                    if DEBUG:
                        print(f'ind check passed: {ind_check}')
                elif s_fal.check() == z3.unsat:
                    print(f"falsifying predicate found! \n{z3.And(*curr_bound_z3)}")
                    num_smt_queries += 2
                    if DEBUG:
                        print(f'fals check: {fal_check}')
                    break
                else:
                    num_smt_queries += 2
                    # split the bounds
                    if DEBUG:
                        print(f'splitting {curr_bound.cpu().detach().numpy()}')
                    tcb = curr_bound.transpose(0, 1)
                    sub_bound_x1, sub_bound_x2 = split_bound(tcb[0])
                    sub_bound_y1, sub_bound_y2 = split_bound(tcb[1])

                    for x, y in product([sub_bound_x1, sub_bound_x2], [sub_bound_y1, sub_bound_y2]):
                        lb = torch.Tensor([x[0][0], y[0][0]])
                        ub = torch.Tensor([x[0][1], y[0][1]])
                        c = torch.Tensor([x[1], y[1]]).unsqueeze(0)
                        bound = torch.cat([lb.unsqueeze(0), ub.unsqueeze(0)], dim=0)

                        # print(x, y)
                        # print(lb, ub, c, bound)
                        queue_to_verify.append((bound, c))
                    
                    num_split += 1
                
            if num_split >= MAX_SPLIT:
                print("!MAX SPLIT REACHED!")

        end_time = time.time()
        duration = end_time - start_time
        print(f"TIME: {duration}s")
        print(f"NUM SPLIT: {num_split}")
        print(f"NUM CLAUSES: {len(bridge_list)}")
        print(f"NUM SMT QUERIES: {num_smt_queries}")
        print(f"NUM NNV QUERIES: {num_nnv_queries}")
        print()