import os
import numpy as np
import pandas as pd
import QLFunction
import torch

quantiles = list(np.linspace(0.01, 0.99, 99))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QL:
    def __init__(self, month_n):
        super(QL, self).__init__()
        #read the file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.join(dir_path,"../../Data/GEF/Load/Task ")
        task_path = base_path + str(month_n)
        benchmark_path = os.path.join(task_path, "L" + str(month_n) + "-benchmark.csv")
        print(benchmark_path)

        #read benmark csv
        benchmark = pd.read_csv(benchmark_path)
        self.b_values = torch.from_numpy(np.array(benchmark.values[:, 3], dtype=float)).to(device)

        #read next month's true value, or labels
        task_path = base_path + str(month_n+1)
        train_path = os.path.join(task_path, task_path, "L" + str(month_n+1) + "-train.csv")
        train = pd.read_csv(train_path)
        self.t_values = torch.from_numpy(np.array(train.values[:, 2], dtype=float)).to(device)


    def compute(self, preds):
        loss_func = QLFunction.QuantileLossFunction(quantiles)
        print("quantile loss between benchmark and true values for one month" + loss_func(self.b_values, self.t_values))







class main:
    qloss = QL(1)
    qloss.compute(1)

#debug
if __name__=="__main":
    main()