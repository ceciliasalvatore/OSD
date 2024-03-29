import os
import time
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from gosdt import GOSDT
from SupervisedDiscretization.discretizer import FCCA, TotalDiscretizer, QuantileDiscretizer
from sklearn.ensemble import GradientBoostingClassifier
from OSD.OSD import DYNACON_SD

if __name__ == '__main__':
    results_dir = "results"
    datasets = ['boston', 'ionosphere', 'magic', 'particle', 'vehicle']

    train_fcca = True
    train_dynacon = True

    regularization = 10
    depth = 4
    print(f'dataset,index,strategy,param,time_discr,T,#tao,compression_tr,compression_ts,inconsistency_tr,inconsistency_ts,time_gosdt,accuracy_tr,accuracy_ts', file=open(f'{results_dir}/results.txt', 'a'))

    for dataset in datasets:
        #with open(f"{results_dir}/{dataset}_log.txt", 'a') as sys.stdout:
            data = pd.read_csv(f'data/{dataset}.csv')

            y = data[data.columns[-1]]
            x = data[data.columns[:-1]]

            M = 2000
            if len(x) < M:
                folds = StratifiedKFold(n_splits=5, random_state=101, shuffle=True)
                splits = folds.split(x, y)
                base_discretizer = TotalDiscretizer()
            else:
                base_discretizer = QuantileDiscretizer(n=100)
                x_tr = x.sample(n=M)
                x_ts = x.drop(x_tr.index,axis=0)
                splits = [(x_tr.index,x_ts.index)]

            for i, (train_index, test_index) in enumerate(splits):
                name = f"{results_dir}/{dataset}_{i}"
                if os.path.exists(f'{name}_tr.csv'):
                    data_tr = pd.read_csv(f'{name}_tr.csv', index_col=0)
                    data_ts = pd.read_csv(f'{name}_ts.csv', index_col=0)
                    y_tr = data_tr[data_tr.columns[-1]]
                    x_tr = data_tr[data_tr.columns[:-1]]
                    y_ts = data_ts[data_ts.columns[-1]]
                    x_ts = data_ts[data_ts.columns[:-1]]
                else:
                    x_tr = x.iloc[train_index]
                    y_tr = y.iloc[train_index]
                    x_ts = x.iloc[test_index]
                    y_ts = y.iloc[test_index]

                    pd.concat((x_tr, y_tr), axis=1).to_csv(f'{name}_tr.csv')
                    pd.concat((x_ts, y_ts), axis=1).to_csv(f'{name}_ts.csv')

                if train_fcca:
                    print("FCCA")
                    discretizer = FCCA(estimator=GradientBoostingClassifier(max_depth=1, n_estimators=100))
                    t0 = time.time()
                    discretizer.fit(x_tr, y_tr)
                    t_discr = time.time()-t0

                    for Q in [0, 0.6, 0.7, 0.8, 0.9, 0.95]:
                        print(f"FCCA {Q}")
                        tao = discretizer.selectThresholds(Q)
                        tao.to_csv(f"{name}_fcca_{Q}.txt", index=False)

                        x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr,tao)
                        x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts,tao)

                        t0 = time.time()
                        gosdt = GOSDT({'regularization': regularization / len(x_tr_discr), 'depth_budget': depth, 'time_limit': 1 * 60 * 60, 'verbose': True})
                        gosdt.fit(x_tr_discr, y_tr_discr)
                        t_gosdt = time.time() - t0
                        print(gosdt.tree)

                        print(f'{dataset},{i},fcca,{Q},{t_discr},0,{len(tao)},{discretizer.compression_rate(x_tr, y_tr,tao)},{discretizer.compression_rate(x_ts, y_ts,tao)},{discretizer.inconsistency_rate(x_tr, y_tr,tao)},{discretizer.inconsistency_rate(x_ts, y_ts,tao)},{t_gosdt},{accuracy_score(y_tr_discr,gosdt.predict(x_tr_discr))},{accuracy_score(y_ts_discr,gosdt.predict(x_ts_discr))}',file=open(f'{results_dir}/results.txt', 'a'))
                        sys.stdout.flush()

                if train_dynacon:
                    for rho in [0.8, 0.6, 0.4, 0.2]:
                        for lambda_ in [100, 10, 1, 0.1, 0.01]:
                            print(f"DYNACON {rho}-{lambda_}")
                            discretizer = DYNACON_SD(rho=rho, lambda_=lambda_, slide=0.01, window=0.05, discretizer=base_discretizer)
                            t0 = time.time()
                            discretizer.fit(x_tr, y_tr)
                            t_discr = time.time() - t0
                            discretizer.tao.to_csv(f"{name}_dynacon_sd_{rho}_{lambda_}.txt", index=False)

                            x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr)
                            x_ts_discr, y_ts_discr = discretizer.transform(x_ts, y_ts)

                            try:
                                t0 = time.time()
                                gosdt = GOSDT({'regularization': regularization / len(x_tr_discr), 'depth_budget': depth, 'time_limit': 1 * 60 * 60, 'verbose': True})
                                gosdt.fit(x_tr_discr, y_tr_discr)
                                t_gosdt = time.time() - t0
                                print(gosdt.tree)

                                print(f'{dataset},{i},dynacon_sd,{rho}-{lambda_},{t_discr},{len(discretizer.discretizer.tao)},{len(discretizer.tao)},{discretizer.compression_rate(x_tr, y_tr)},{discretizer.compression_rate(x_ts, y_ts)},{discretizer.inconsistency_rate(x_tr, y_tr)},{discretizer.inconsistency_rate(x_ts, y_ts)},{t_gosdt},{accuracy_score(y_tr_discr,gosdt.predict(x_tr_discr))},{accuracy_score(y_ts_discr,gosdt.predict(x_ts_discr))}',file=open(f'{results_dir}/results.txt', 'a'))
                                sys.stdout.flush()
                            except:
                                print(f'{dataset},{i},dynacon_sd,{rho}-{lambda_},{t_discr},{len(discretizer.discretizer.tao)},{len(discretizer.tao)},{discretizer.compression_rate(x_tr, y_tr)},{discretizer.compression_rate(x_ts, y_ts)},{discretizer.inconsistency_rate(x_tr, y_tr)},{discretizer.inconsistency_rate(x_ts, y_ts)},0,0,0',file=open(f'{results_dir}/results.txt', 'a'))
