import gc



import pandas as pd

import os

import argparse


from utils.utils import enable_reproducible_results, simulate_scenarios
from dataloaders import dataset_loader


from model.wgf_imp import NeuralGradFlowImputer
import torch
torch.set_default_tensor_type('torch.DoubleTensor')


# basic params
parser = argparse.ArgumentParser(prog='Basic')
parser.add_argument('--model', default='WassGF')
parser.add_argument('--seed', default=4, type=int)
parser.add_argument('--outpath', default='./results/flowImpResults')
parser.add_argument('--verbose', default=1)

parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--n_epochs', default=150, type=int)
parser.add_argument('--n_pairs', default=10, type=int)
parser.add_argument('--noise', default=1e-4, type=float)
parser.add_argument('--numItermax', default=1000, type=int)
parser.add_argument('--stopThr', default=1e-3, type=float)
parser.add_argument('--alpha', default=0, type=float)

parser.add_argument('--bandwidth', default=0.5, type=float)
parser.add_argument('--entropy_reg', default=10.0, type=float)
parser.add_argument('--score_net_epoch', default=200, type=int)
parser.add_argument('--iter_time', default=2, type=int)


parser.add_argument('--initializer', default='mean', type=str)
parser.add_argument('--mlp_hidden', default='[128, 128]', type=str)

parser.add_argument('--ode_step', default=1.0e-1, type=float)
parser.add_argument('--score_lr', default=1.0e-3, type=float)
parser.add_argument('--dataset_name', default="blood_transfusion", type=str)


# parse_args operations
args = parser.parse_args()


# get the dataset
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")

ground_truth = dataset_loader(args.dataset_name)







SCENARIO = [
    "MAR",
    "MNAR",
    # "MCAR" # optional
]
P_MISS = [
    0.3,
]

# setup random seed
enable_reproducible_results(args.seed)
print(f"we are running at: {args.seed}")
X = ground_truth
imputation_scenarios = simulate_scenarios(X, mechanisms=SCENARIO, percentages=P_MISS)

result_df = pd.DataFrame()
for scenario in SCENARIO:
    for p_miss in P_MISS:

        model = NeuralGradFlowImputer(entropy_reg=args.entropy_reg, bandwidth=args.bandwidth,
                                      score_net_epoch=args.score_net_epoch, niter=args.iter_time,
                                      initializer=None, mlp_hidden=eval(args.mlp_hidden), lr=args.ode_step,
                                      score_net_lr=args.score_lr)

        enable_reproducible_results(args.seed)
        x, x_miss, mask = imputation_scenarios[scenario][p_miss]
        _, result_list = model.fit_transform(x_miss.copy().values, verbose=True, report_interval=1, X_true=x.copy().values)
        # print(result_list)
        result_list = [pd.DataFrame(dict_idx, index=[0]) for dict_idx in result_list]
        temp_result_df = pd.concat(result_list, axis=0)
        temp_result_df["missing"] = scenario
        temp_result_df["dataset_name"] = args.dataset_name
        temp_result_df["seed"] = args.seed
        temp_result_df["p_miss"] = p_miss
        result_df = pd.concat([result_df, temp_result_df], axis=0)
        del x, x_miss, mask, model
        gc.collect()
        torch.cuda.empty_cache()
        # result_list = [pd.DataFrame({})]


# print(result_df)
        # _, result_list = NeuralGradFlowImputer.fit_transform(x)
result_df = result_df[result_df["interval"] == int(args.iter_time - 1)].reset_index(drop=True)
os.makedirs(args.outpath) if not os.path.exists(args.outpath) else None
csv_name = (f"model_{args.model}_data_{args.dataset_name}_seed_{args.seed}_"
            f"_ode_step_{args.lr}_bandwidth_{args.bandwidth}_reg_{args.entropy_reg}"
            f"_score_{eval(args.mlp_hidden)[0]}_epoch_{args.score_net_epoch}.csv")
print(result_df)
result_df.to_csv(os.path.join(args.outpath, csv_name), index=None)







