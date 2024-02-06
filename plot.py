import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    datasets = ['boston','ionosphere', 'magic', 'particle']
    res = pd.read_csv('results/results.txt')

    strategies = ['fcca', 'dynacon_sd']
    x_labels = {'fcca':'Q', 'dynacon_sd':'lambda','indigo_sd':'lambda'}
    font_size = 14

    for dataset in datasets:
        res_dataset = res.query(f"dataset=='{dataset}'").groupby(['strategy','param']).mean()
        res_dataset['accuracy_tr'] = res_dataset['accuracy_tr'] * 100
        res_dataset['accuracy_ts'] = res_dataset['accuracy_ts'] * 100
        #res_dataset.round(2).to_csv(f'results_tesi/{dataset}.txt')

        # FCCA
        res_fcca = res_dataset.reset_index().query(f"strategy=='fcca'")

        fig, ax = plt.subplots(1,1, figsize=(8, 5))
        color = 'firebrick'
        ax.set_title(f"Compression and Inconsistency of FCCA on {dataset}", fontsize=font_size)
        ax.set_xlabel('Q', fontsize=font_size)
        ax.set_ylabel('compression', color=color, fontsize=font_size)
        ax.plot(np.arange(len(res_fcca)), res_fcca['compression_ts'], marker='o', linestyle='dashed', color=color, linewidth=5, markersize=10)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, np.ceil(res_dataset['compression_ts'].max() * 10) / 10)
        ax.set_xticks(np.arange(len(res_fcca)), res_fcca.param.to_list())
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'darkcyan'
        ax2.set_xlabel('Q', fontsize=font_size)
        ax2.set_ylabel('inconsistency', color=color, fontsize=font_size)
        ax2.plot(np.arange(len(res_fcca)), res_fcca['inconsistency_ts'], marker='s', linestyle='dashed',  color=color, linewidth=5, markersize=10)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, np.ceil(res_dataset['inconsistency_ts'].max() * 10) / 10)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'results/{dataset}_fcca_compression_inconsistency.png')

        fig, ax = plt.subplots(1,1, figsize=(8, 5))
        color = 'darkgreen'
        ax.set_title(f"Accuracy of FCCA on {dataset}", fontsize=font_size)
        ax.set_xlabel('Q', fontsize=font_size)
        ax.set_ylabel('accuracy', color=color, fontsize=font_size)
        ax.plot(np.arange(len(res_fcca)), res_fcca['accuracy_ts'], marker='o', linestyle='dashed', color=color, linewidth=5, markersize=10)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(np.floor(res_dataset['accuracy_ts'].min() * 10) / 10,
                        np.ceil(res_dataset['accuracy_ts'].max() * 10) / 10)
        ax.set_xticks(np.arange(len(res_fcca)), res_fcca.param.to_list())

        plt.savefig(f'results/{dataset}_fcca_accuracy.png')

        # DYNACON_SD
        res_dynacon = res_dataset.reset_index().query(f"strategy=='dynacon_sd'")
        res_dynacon['rho'] = res_dynacon['param'].apply(lambda x: x.split('-')[0])
        res_dynacon['lambda'] = res_dynacon['param'].apply(lambda x: x.split('-')[1])

        for rho in [0.8, 0.6, 0.4, 0.2]:
            res_dynacon_rho = res_dynacon.query(f"rho=='{rho}'")
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            color = 'firebrick'
            ax.set_title(f"Compression and Inconsistency of DYNACON_SD with rho={rho} on {dataset}", fontsize=font_size)
            ax.set_xlabel('lambda', fontsize=font_size)
            ax.set_ylabel('compression', color=color, fontsize=font_size)
            ax.plot(np.arange(len(res_dynacon_rho)), res_dynacon_rho['compression_ts'], marker='o', linestyle='dashed', color=color, linewidth=5, markersize=10)
            ax.tick_params(axis='y', labelcolor=color)
            ax.set_ylim(0, np.ceil(res_dataset['compression_ts'].max() * 10) / 10)
            ax.set_xticks(np.arange(len(res_dynacon_rho)), res_dynacon_rho['lambda'].to_list())
            ax.invert_xaxis()
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'darkcyan'
            ax2.set_xlabel('Q', fontsize=font_size)
            ax2.set_ylabel('inconsistency', color=color, fontsize=font_size)
            ax2.plot(np.arange(len(res_dynacon_rho)), res_dynacon_rho['inconsistency_ts'], marker='s', linestyle='dashed',  color=color, linewidth=5, markersize=10)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, np.ceil(res_dataset['inconsistency_ts'].max() * 10) / 10)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(f'results/{dataset}_dynacon_{rho}_compression_inconsistency.png')

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            color = 'darkgreen'
            ax.set_title(f"Accuracy of DYNACON_SD with rho={rho} on {dataset}", fontsize=font_size)
            ax.set_xlabel('lambda', fontsize=font_size)
            ax.set_ylabel('accuracy', color=color, fontsize=font_size)
            ax.plot(np.arange(len(res_dynacon_rho)), res_dynacon_rho['accuracy_ts'], marker='o', linestyle='dashed', color=color, linewidth=5, markersize=10)
            ax.tick_params(axis='y', labelcolor=color)
            ax.set_ylim(np.floor(res_dataset['accuracy_ts'].min() * 10) / 10,
                        np.ceil(res_dataset['accuracy_ts'].max() * 10) / 10)
            ax.set_xticks(np.arange(len(res_dynacon_rho)), res_dynacon_rho['lambda'].to_list())
            ax.invert_xaxis()

            plt.savefig(f'results/{dataset}_dynacon_{rho}_accuracy.png')
