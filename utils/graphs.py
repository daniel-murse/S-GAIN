# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graphing functions for S-GAIN:

(1) plot_rmse: plot the RMSE
(2) plot_imputation_time: plot the imputation time
(3) plot_memory_usage: plot the memory usage
(4) plot_energy_consumption: plot the energy consumption
(5) plot_sparsity: plot the sparsity
(6) plot_flops: plot the flops
(7) plot_loss: plot the loss
(8) plot_all: plot all graphs
"""

from datetime import timedelta

import matplotlib.pyplot as plt


def plot_rmse(filepath, log):
    """Load and plot the RMSE.

    :param filepath: the filepath for the RMSE graph
    :param log: the RMSE log
    """

    # Plot parameters
    plt.figure(figsize=(12.8, 4.8))
    plt.title('RMSE per epoch')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(title='Final')

    # Todo: plot the graph (use the legend to display the final)

    plt.savefig(filepath, format='png')


def plot_imputation_time(filepath, log):
    """Load and plot the imputation time.

    :param filepath: the filepath for the imputation time graph
    :param log: the imputation time log
    """

    # New plot
    plt.figure(figsize=(12.8, 4.8))
    plt.plot(log, label=f'{timedelta(seconds=round(sum(log)))}')
    len_log = len(log)

    # Plot parameters
    plt.title('Imputation time per epoch')
    plt.ylabel('Time (in seconds)')
    plt.xlabel('Epoch')
    plt.legend(title='Total')
    plt.xlim(-len_log * 0.01, len_log * 1.01)

    # Save plot
    plt.savefig(filepath, format='png')


def plot_memory_usage(filepath, log):
    """Load and plot the memory usage.

    :param filepath: the filepath for the memory usage graph
    :param log: the memory usage log
    """

    # Plot parameters
    plt.figure(figsize=(12.8, 4.8))
    plt.title('Memory usage per epoch')
    plt.ylabel('memory usage (in MB)')
    plt.xlabel('epoch')
    plt.legend(title='Total')

    # Todo: plot the graph (use the legend to display the total)

    plt.savefig(filepath, format='png')


def plot_energy_consumption(filepath, log):
    """Load and plot the energy consumption.

    :param filepath: the filepath for the energy consumption graph
    :param log: the energy consumption log
    """

    # Plot parameters
    plt.figure(figsize=(12.8, 4.8))
    plt.title('Energy consumption per epoch')
    plt.ylabel('energy consumption (in joule)')
    plt.xlabel('epoch')
    plt.legend(title='Total')

    # Todo: plot the graph (use the legend to display the total)

    plt.savefig(filepath, format='png')


def plot_sparsity(filepath, log, log_G, log_G_W1, log_G_W2, log_G_W3, log_D, log_D_W1, log_D_W2, log_D_W3):
    """Load and plot the sparsity.

    :param filepath: the filepath for the sparsity graph
    :param log: the sparsity log for S-GAIN (total)
    :param log_G: the sparsity log for the Generator
    :param log_G_W1: the sparsity log for the first layer of the Generator
    :param log_G_W2: the sparsity log for the second layer of the Generator
    :param log_G_W3: sparsity log for the third layer of the Generator
    :param log_D: the sparsity log for the Discriminator
    :param log_D_W1: the sparsity log for the first layer of the Discriminator
    :param log_D_W2: the sparsity log for the second layer of the Discriminator
    :param log_D_W3: the sparsity log for the third layer of the Discriminator
    """

    # Plot parameters
    plt.figure(figsize=(12.8, 4.8))
    plt.title('Sparsity per epoch')
    plt.ylabel('Sparsity')
    plt.xlabel('epoch')
    plt.legend(title='Final')

    # Todo: plot the graph (use the legend to display the final for total, G, D, layers)

    plt.savefig(filepath, format='png')


def plot_flops(filepath, log, log_G, log_D):
    """Load and plot the FLOPs.

    :param filepath: the filepath for the FLOPs graph
    :param log: the FLOPs log for S-GAIN (total)
    :param log_G: the FLOPs log for the Generator
    :param log_D: the FLOPs log for the Discriminator
    """

    # Plot parameters
    plt.figure(figsize=(12.8, 4.8))
    plt.title('FLOPs per epoch')
    plt.ylabel('FLOPs')
    plt.xlabel('epoch')
    plt.legend(title='Total')

    # Todo: plot the graph (use the legend to display the total, G, D)

    plt.savefig(filepath, format='png')


def plot_loss(filepath, log_G, log_D, log_MSE):
    """Load and plot the loss.

    :param filepath: the filepath for the loss graph
    :param log_G: the loss log for the Generator (cross entropy)
    :param log_D: the loss log for the Discriminator (cross entropy)
    :param log_MSE: the loss log (MSE)
    """

    # New plot
    fig, (ax_ce, ax_mse) = plt.subplots(2, figsize=(12.8, 9.6))
    ax_ce.plot(log_G, label='Generator loss')
    ax_ce.plot(log_D, label='Discriminator loss')
    ax_mse.plot(log_MSE, label='MSE')

    len_logs  = max(len(log_G), len(log_D), len(log_MSE))

    # Cross Entropy parameters
    ax_ce.title.set_text('Learning curves')
    ax_ce.set_ylabel('Cross Entropy')
    ax_ce.set_xlabel('Epoch')

    ax_ce.set_xlim(-len_logs * 0.01, len_logs * 1.01)
    ax_ce.legend()
    ax_ce.grid(True)

    # MSE parameters
    ax_mse.title.set_text('Learning curves')
    ax_mse.set_ylabel('MSE loss')
    ax_mse.set_xlabel('Epoch')
    ax_mse.set_xlim(-len_logs * 0.01, len_logs * 1.01)
    ax_mse.legend()
    ax_mse.grid(True)

    # Save plot
    plt.savefig(filepath, format='png')


def plot_all(filepath_rmse, filepath_imputation_time, filepath_memory_usage, filepath_energy_consumption,
             filepath_sparsity, filepath_flops, filepath_loss, logs):
    """Load the log file and plot all.

    :param filepath_rmse: the filepath for the RMSE graph
    :param filepath_imputation_time: the filepath for the imputation time graph
    :param filepath_memory_usage: the filepath for the memory usage graph
    :param filepath_energy_consumption: the filepath for the energy consumption graph
    :param filepath_sparsity: the filepath for the sparsity graph
    :param filepath_flops: the filepath for the flops graph
    :param filepath_loss: the filepath for the loss graph
    :param logs: a list of all logs
    """

    # Get all the logs from the list
    log_rmse, log_imputation_time, log_memory_usage, log_energy_consumption, log_sparsity, \
        log_sparsity_G, log_sparsity_G_W1, log_sparsity_G_W2, log_sparsity_G_W3, \
        log_sparsity_D, log_sparsity_D_W1, log_sparsity_D_W2, log_sparsity_D_W3, \
        log_flops, log_flops_G, log_flops_D, log_loss_G, log_loss_D, log_loss_MSE = logs

    # Plot all the logs
    # plot_rmse(filepath_rmse, log_rmse)
    plot_imputation_time(filepath_imputation_time, log_imputation_time)
    # plot_memory_usage(filepath_memory_usage, log_memory_usage)
    # plot_energy_consumption(filepath_energy_consumption, log_energy_consumption)
    # plot_sparsity(filepath_sparsity, log_sparsity, log_sparsity_G, log_sparsity_G_W1, log_sparsity_G_W2,
    #               log_sparsity_G_W3, log_sparsity_D, log_sparsity_D_W1, log_sparsity_D_W2, log_sparsity_D_W3)
    # plot_flops(filepath_flops, log_flops, log_flops_G, log_flops_D)
    plot_loss(filepath_loss, log_loss_G, log_loss_D, log_loss_MSE)