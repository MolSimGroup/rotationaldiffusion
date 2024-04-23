import numpy as np
import matplotlib.pyplot as plt


def create_plot_for_Q(figsize=(10, 4), constrained_layout=True,
                      sharex=True, sharey='row', **kwargs):
    fig, axs = plt.subplots(2, 3, figsize=figsize,
                            constrained_layout=constrained_layout,
                            sharex=sharex, sharey=sharey, **kwargs)
    return fig, axs


def plot_Q(fig, axs, lag_times, Q, labels=True, xlabel='$\\tau$ / ns',
           ylabel='Q$_{ii}$', **kwargs):
    if labels:
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)

    Q = np.moveaxis(Q, 0, -1)
    for i in range(3):
        j, k = max(i, 1) - 1, min(i, 1) + 1
        axs[0, i].plot(lag_times, Q[i, i], **kwargs)
        axs[1, i].plot(lag_times, Q[j, k], **kwargs)

        if labels:
            axs[0, i].set_title(f"Q$_{{{i}{i}}}$")
            axs[1, i].set_title(f"Q$_{{{j}{k}}}$")
    return


def plot_instantaneous_D_PAF(lag_times, D, PAF, figsize=(8, 5), sharex=True,
                             constrained_layout=True, xlabel='$\\tau$ / ns',
                             ylabel0='D / ns$^{-1}$', ylabel1='cos $\\alpha$',
                             **kwargs):
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=sharex,
                            constrained_layout=constrained_layout, **kwargs)
    axs[1].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel0)
    axs[1].set_ylabel(ylabel1)

    for i, axis in enumerate('xyz'):
        axs[0].plot(lag_times, D[..., i], label=f"{axis}-axis")
        cos = np.abs(np.dot(PAF[:, i], PAF[0, i]))
        axs[1].plot(lag_times, cos)
    axs[0].legend(loc='lower center', ncol=3)
    return fig, axs