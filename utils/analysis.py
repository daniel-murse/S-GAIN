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

"""Analysis functions for S-GAIN:

Todo properly name the graphs, legends, labels, etc

(1) compile_metrics: compile the metrics of the provided experiments
(2) plot_rmse: plot the RMSE of the provided experiments
(3) plot_success_rate: plot the success rate of the provided experiments
"""

from os import mkdir
from os.path import isdir
from matplotlib import ticker, container

from utils.graphs2 import get_sizing, plot_info

import matplotlib.pyplot as plt

# Groupings
exp = 'Experiments'
d_mr_mm_s = ['dataset', 'miss_rate', 'miss_modality', 'seed']
bs_hr_a_i = ['batch_size', 'hint_rate', 'alpha', 'iterations']
gs = ['generator_sparsity']
gm = ['generator_modality']
ds = ['discriminator_sparsity']
dm = ['discriminator_modality']


def compile_metrics(experiments, save=None, folder=None, verbose=False):
    """Compile the metrics of the provided experiments.
    Metrics: RMSE mean, std and improvement, and successes, total runs and success rate

    :param experiments: a Pandas DataFrame with the experiments to compile
    :param save: whether to save the compiled metrics
    :param folder: the folder to save the compiled metrics to
    :param verbose: enable verbose output to console

    :return:
    - exps: a Pandas DataFrame of the compiled metrics
    """

    if verbose: print('Compiling metrics...')

    # Calculate mean, std, successes, total_runs and success_rate
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    exps = exps.groupby(d_mr_mm_s + bs_hr_a_i + gs + gm + ds + dm, as_index=False).agg(['mean', 'std', 'count', 'size'])
    exps.columns = exps.columns.get_level_values(0) + exps.columns.get_level_values(1)
    exps.rename(columns={'rmsemean': 'rmse_mean', 'rmsestd': 'rmse_std', 'rmsecount': 'successes',
                         'rmsesize': 'total_runs'}, inplace=True)
    exps['success_rate'] = exps['successes'] / exps['total_runs']

    # Calculate RMSE improvement compared to dense
    dense = exps.loc[(exps['generator_sparsity'] == 0) & (exps['discriminator_sparsity'] == 0)]
    if not dense.empty:
        for d, mr, mm, s, bs, hr, a, i, _, _, _, _, rmse_mean_dense, _, _, _, _ in dense.values:
            match = exps.loc[(exps['dataset'] == d) & (exps['miss_rate'] == mr) & (exps['miss_modality'] == mm)
                             & (exps['seed'] == s) & (exps['batch_size'] == bs) & (exps['hint_rate'] == hr)
                             & (exps['alpha'] == a) & (exps['iterations'] == i)]
            exps.loc[match.index, 'rmse_improvement'] = 1 / (match['rmse_mean'] / rmse_mean_dense) - 1
        exps.insert(len(exps.columns) - 4, 'rmse_improvement', exps.pop('rmse_improvement'))  # Move column
        exps['rmse_improvement'] = exps['rmse_improvement'].round(3)  # Rounding

    # Rounding
    exps['rmse_mean'] = exps['rmse_mean'].round(4)
    exps['rmse_std'] = exps['rmse_std'].round(7)
    exps['success_rate'] = exps['success_rate'].round(2)

    # Save the metrics
    if save:
        if verbose: print('Saving compiled metrics...')
        if not isdir(folder): mkdir(folder)
        exps.to_csv(f'{folder}/metrics.csv', index=False)

    return exps


def plot_rmse(experiments, save=False, folder='analysis', sys_info=None, verbose=False):
    """Plot the RMSE of the provided experiments.

    :param experiments: a Pandas DataFrame with the experiments to compile
    :param save: whether to save the compiled metrics
    :param folder: the folder to save the compiled metrics to
    :param sys_info: the system info (in print ready format)
    :param verbose: enable verbose output to console
    """

    def subplot(ax, M_rmse_mean_std, M, legend_ax, sparsity='all', modality='settings'):
        """Create a subplot per model setting.

        :param ax: the subplot to write to
        :param M_rmse_mean_std: the rmse mean and std group
        :param M: the model this belongs to (Generator or Discriminator)
        :param legend_ax: the subplot to write the legend to
        :param sparsity: the sparsity of the other model (optional)
        :param modality: the modality of the other model (optional)
        """

        # Set parameters
        if legend_ax is None: legend_ax = ax
        if sparsity != 'all': sparsity = f'{int(sparsity * 100)}%'
        if M in ('G', 'generator'):
            ax.title.set_text(f'Discriminator: {sparsity} {modality}')
            subtitle = 'Generator'
            m = ['generator_modality']
            sparsity = 'generator_sparsity'
        else:  # ('D', 'discriminator')
            ax.title.set_text(f'Generator: {sparsity} {modality}')
            subtitle = 'Discriminator'
            m = ['discriminator_modality']
            sparsity = 'discriminator_sparsity'

        # Only show batch_size, hint_rate, alpha and iterations if different settings were tested
        show_bs_hr_a_i = False if M_rmse_mean_std.groupby(bs_hr_a_i).ngroups == 1 else True

        # Group by batch_size, hint_rate, alpha, iterations, modality
        for (batch_size, hint_rate, alpha, iterations, modality), bs_hr_a_i_m_group \
                in M_rmse_mean_std.groupby(bs_hr_a_i + m):
            label = f'{batch_size}_{hint_rate}_{alpha}_{iterations}_{modality}' if show_bs_hr_a_i \
                else f'{modality[0].upper()}{modality[1:]}'

            x = bs_hr_a_i_m_group[sparsity]
            y = bs_hr_a_i_m_group['rmse']['mean']
            e = bs_hr_a_i_m_group['rmse']['std']

            # Set color Todo expand for different settings (not only modality)
            color = 'black' if modality == 'dense' \
                else 'tab:orange' if modality == 'random' \
                else 'tab:red' if modality == 'ER' \
                else 'tab:purple' if modality == 'ERRW' \
                else 'tab:green'

            # Add experiment to plot
            if modality == 'dense':
                y_mu = y.iloc[0]
                y_sigma = y_mu + e.iloc[0], y_mu - e.iloc[0]

                ax.axhline(y=y_mu, color=color, linewidth=1, label=label)
                ax.axhline(y=y_sigma[0], color=color, linewidth=1, linestyle='--')
                ax.axhline(y=y_sigma[1], color=color, linewidth=1, linestyle='--')
                ax.axhspan(y_sigma[0], y_sigma[1], facecolor=color, alpha=0.2)

            else:
                ax.errorbar(x, y, e, color=color, marker='o', markersize=3, capsize=5, elinewidth=1, label=label)
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.2)

            # Plot parameters
            ax.title.set_size(16)
            ax.grid(True)
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1., decimals=0))  # Format sparsity as percentage
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))  # Format RMSE to 4 decimals
            ax.set_xlabel(f'{subtitle} sparsity', size=13)
            ax.set_ylabel('RMSE ± 1 SD', size=13)  # Todo CI
            ax.tick_params(labelsize=12)

        # Update the legend only on the first plot
        if not legend_ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]  # Remove error bars
            lgnd = legend_ax.legend(handles, labels, fontsize=12, loc='upper left')
            lgnd_title = f'Batch size, hint rate, alpha,\niterations and {subtitle.lower()} modality' \
                if show_bs_hr_a_i else f'{subtitle} modality'
            lgnd.set_title(lgnd_title, prop={'size': 13})

    if verbose: print(f'Plotting RMSE...')

    # Group by dataset, miss_rate, miss_modality and seed
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    for (dataset, miss_rate, miss_modality, seed), d_mr_mm_s_group in exps.groupby(d_mr_mm_s):

        title = f'{dataset}_{miss_rate}_{miss_modality}_{hex(seed)}'
        dataset = f'Dataset: {dataset}'
        miss_rate = f'Miss rate: {int(miss_rate * 100)}%'
        miss_modality = f'Miss modality: {miss_modality}'
        seed = f'Seed: {hex(seed)}'
        text = [exp, dataset, miss_rate, miss_modality, seed, ' ']
        if sys_info: text += sys_info

        if verbose: print(f'{title}')
        d_mr_mm_s_group.drop(d_mr_mm_s, axis='columns', inplace=True)

        # Get the tested sparsities and modalities, and the number of needed rows for the graph
        Gsm = d_mr_mm_s_group.drop_duplicates(subset=['generator_sparsity', 'generator_modality'])
        Gsm = Gsm[['generator_sparsity', 'generator_modality']].values.tolist()
        Gsm.sort()
        nGsm = len(Gsm)

        Dsm = d_mr_mm_s_group.drop_duplicates(subset=['discriminator_sparsity', 'discriminator_modality'])
        Dsm = Dsm[['discriminator_sparsity', 'discriminator_modality']].values.tolist()
        Dsm.sort()
        nDsm = len(Dsm)

        nrows = max(nGsm, nDsm) + 1

        if nDsm == 1 or nGsm == 1:
            # Plot parameters
            width, height, left, right, top, bottom, wspace, _ = get_sizing(3, 1, 6.4, 4.8)
            fig, axs = plt.subplots(1, 3, figsize=(width, height))
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)

            # Show the influence of different settings for the Generator
            if nDsm == 1:
                G_group = d_mr_mm_s_group.drop(ds + dm, axis='columns')
                G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['mean', 'std'])
                subplot(axs[0], G_rmse_mean_std, 'G', axs[1], Dsm[0][0], Dsm[0][1])
                axs[1].set_axis_off()
                axs[2].set_axis_off()

            # Show the influence of different settings for the Discriminator
            else:  # nGsm == 1
                D_group = d_mr_mm_s_group.drop(gs + gm, axis='columns')
                D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['mean', 'std'])
                subplot(axs[0], D_rmse_mean_std, 'D', axs[1], Gsm[0][0], Gsm[0][1])
                axs[1].set_axis_off()
                axs[2].set_axis_off()

            # Plot system information
            plot_info(axs[2], text)

        else:  # Multiple settings used for both the Generator and Discriminator
            # Plot parameters
            width, height, left, right, top, bottom, wspace, hspace = get_sizing(2, nrows, 6.4, 4.8)
            fig, axs = plt.subplots(nrows, 2, figsize=(width, height))
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

            # Remove unused subplots and use the first and second empty subplot for the legend and system information
            info_ax, legend_ax, axs_set_axis_off = None, None, []
            for i in range(nGsm):
                if i >= nDsm:
                    axs_set_axis_off.append((i + 1, 0))
                    if not info_ax:
                        info_ax = (i + 1, 0)
                    elif not legend_ax:
                        legend_ax = (i + 1, 0)

            for i in range(nDsm):
                if i >= nGsm:
                    axs_set_axis_off.append((i + 1, 1))
                    if not info_ax:
                        info_ax = (i + 1, 1)
                    elif not legend_ax:
                        legend_ax = (i + 1, 1)

            if not info_ax or not legend_ax:
                width, height, left, right, top, bottom, wspace, hspace = get_sizing(3, nrows, 6.4, 4.8)
                fig, axs = plt.subplots(nrows, 3, figsize=(width, height))
                plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

                for i in range(nrows): axs_set_axis_off.append((i, 2))
                if not info_ax:
                    info_ax, legend_ax = (0, 2), (1, 2)
                else:  # not legend_ax
                    legend_ax = (0, 2)

            for coord in axs_set_axis_off: axs[coord[0], coord[1]].set_axis_off()
            legend_ax = axs[legend_ax[0], legend_ax[1]]
            info_ax = axs[info_ax[0], info_ax[1]]

            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            G_group = d_mr_mm_s_group.drop(ds + dm, axis='columns')
            G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['mean', 'std'])
            subplot(axs[0, 0], G_rmse_mean_std, 'G', legend_ax)

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            D_group = d_mr_mm_s_group.drop(gs + gm, axis='columns')
            D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['mean', 'std'])
            subplot(axs[0, 1], D_rmse_mean_std, 'D', legend_ax)

            # Show the influence of different settings for the Generator for different Discriminator settings
            for i in range(nDsm):
                G_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['discriminator_sparsity'] == Dsm[i][0])
                    & (d_mr_mm_s_group['discriminator_modality'] == Dsm[i][1])
                )
                G_group.drop(ds + dm, axis='columns', inplace=True)
                G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['mean', 'std'])
                subplot(axs[i + 1, 0], G_rmse_mean_std, 'G', legend_ax, Dsm[i][0], Dsm[i][1])

            # Show the influence of different settings for the Discriminator for different Generator settings
            for i in range(nGsm):
                D_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['generator_sparsity'] == Gsm[i][0])
                    & (d_mr_mm_s_group['generator_modality'] == Gsm[i][1])
                )
                D_group.drop(gs + gm, axis='columns', inplace=True)
                D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['mean', 'std'])
                subplot(axs[i + 1, 1], D_rmse_mean_std, 'D', legend_ax, Gsm[i][0], Gsm[i][1])

            # Plot system information
            plot_info(info_ax, text)

        # Plot parameters
        plt.suptitle(title, size=24)

        if save:
            if verbose: print(f'Saving plot...')
            path = f'{folder}/{title}_RMSE.pdf'
            plt.savefig(path, format='pdf', dpi=1200)

        fig.show()


def plot_success_rate(experiments, save=False, folder='analysis', sys_info=None, verbose=False):
    """Plot the success rate of the provided experiments.

    :param experiments: a Pandas DataFrame with the experiments to compile
    :param save: whether to save the compiled metrics
    :param folder: the folder to save the compiled metrics to
    :param sys_info: the system info (in print ready format)
    :param verbose: enable verbose output to console
    """

    def subplot(ax, M_rmse_mean_std, M, legend_ax, sparsity='all', modality='settings', legend_loc='upper right'):
        """Create a subplot per model setting.

        :param ax: the subplot to write to
        :param M_rmse_mean_std: the rmse mean and std group
        :param M: the model this belongs to (Generator or Discriminator)
        :param legend_ax: the subplot to write the legend to
        :param sparsity: the sparsity of the other model (optional)
        :param modality: the modality of the other model (optional)
        :param legend_loc: the location of the legend
        """

        # Set parameters
        if sparsity != 'all': sparsity = f'{int(sparsity * 100)}%'
        if M in ('G', 'generator'):
            ax.title.set_text(f'Discriminator: {sparsity} {modality}')
            subtitle = 'Generator'
            m = ['generator_modality']
            sparsity = 'generator_sparsity'
        else:  # ('D', 'discriminator')
            ax.title.set_text(f'Generator: {sparsity} {modality}')
            subtitle = 'Discriminator'
            m = ['discriminator_modality']
            sparsity = 'discriminator_sparsity'

        # Only show batch_size, hint_rate, alpha and iterations if different settings were tested
        show_bs_hr_a_i = False if M_rmse_mean_std.groupby(bs_hr_a_i).ngroups == 1 else True

        # Group by batch_size, hint_rate, alpha, iterations, modality
        M_rmse_mean_std[m[0]] = M_rmse_mean_std[m[0]].str.lower()  # Prevent ER(K)(RW) before dense
        for (batch_size, hint_rate, alpha, iterations, modality), bs_hr_a_i_m_group \
                in M_rmse_mean_std.groupby(bs_hr_a_i + m):
            x = bs_hr_a_i_m_group[sparsity].map('{:.0%}'.format)
            y = bs_hr_a_i_m_group['success_rate']

            # Set color, re-capitalize modality and correct x Todo expand for different settings (not only modality)
            if modality == 'dense':
                color = 'black'
            elif modality == 'random':
                color = 'tab:orange'
                x = [f' {x} ' for x in x]
            elif modality == 'er':
                modality = 'ER'
                color = 'tab:red'
                x = [f'  {x}  ' for x in x]
            elif modality == 'errw':
                modality = 'ERRW'
                color = 'tab:purple'
                x = [f'   {x}   ' for x in x]
            else:
                color = 'tab:green'
                x = [f'    {x}    ' for x in x]

            label = f'{batch_size}_{hint_rate}_{alpha}_{iterations}_{modality}' if show_bs_hr_a_i \
                else f'{modality[0].upper()}{modality[1:]}'

            # Add experiment to plot
            ax.bar(x, y, color=color, label=label)

            # Plot parameters
            ax.title.set_size(16)
            ax.set_xlabel(f'{subtitle} sparsity', size=13)
            ax.set_ylabel('Success rate', size=13)  # Todo CI
            ax.set_ylim((0, 1.05))
            ax.tick_params(labelsize=12)

        # Update the legend only on the first plot
        if not legend_ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            lgnd = legend_ax.legend(handles, labels, fontsize=12, loc=legend_loc)
            lgnd_title = f'Batch size, hint rate, alpha,\niterations and {subtitle.lower()} modality' \
                if show_bs_hr_a_i else f'{subtitle} modality'
            lgnd.set_title(lgnd_title, prop={'size': 13})

    if verbose: print(f'Plotting success rate...')

    # Group by dataset, miss_rate, miss_modality and seed
    exps = experiments.drop(['index', 'filetype'], axis='columns')
    for (dataset, miss_rate, miss_modality, seed), d_mr_mm_s_group in exps.groupby(d_mr_mm_s):

        title = f'{dataset}_{miss_rate}_{miss_modality}_{hex(seed)}'
        dataset = f'Dataset: {dataset}'
        miss_rate = f'Miss rate: {int(miss_rate * 100)}%'
        miss_modality = f'Miss modality: {miss_modality}'
        seed = f'Seed: {hex(seed)}'
        text = [exp, dataset, miss_rate, miss_modality, seed, ' ']
        if sys_info: text += sys_info

        if verbose: print(f'{title}')
        d_mr_mm_s_group.drop(d_mr_mm_s, axis='columns', inplace=True)

        # Get the tested sparsities and modalities, and the number of needed rows for the graph
        Gsm = d_mr_mm_s_group.drop_duplicates(subset=['generator_sparsity', 'generator_modality'])
        Gsm = Gsm[['generator_sparsity', 'generator_modality']].values.tolist()
        Gsm.sort()
        nGsm = len(Gsm)

        Dsm = d_mr_mm_s_group.drop_duplicates(subset=['discriminator_sparsity', 'discriminator_modality'])
        Dsm = Dsm[['discriminator_sparsity', 'discriminator_modality']].values.tolist()
        Dsm.sort()
        nDsm = len(Dsm)

        nrows = max(nGsm, nDsm) + 1

        if nDsm == 1 or nGsm == 1:  # Only one setting used for the Discriminator
            # Plot parameters
            width, height, left, right, top, bottom, wspace, _ = get_sizing(2, 1, 12.8, 4.8)
            fig, axs = plt.subplots(1, 2, figsize=(width, height))
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)

            # Show the influence of different settings for the Generator
            if nDsm == 1:
                G_group = d_mr_mm_s_group.drop(ds + dm, axis='columns')
                G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['count', 'size'])
                G_rmse_mean_std['success_rate'] = G_rmse_mean_std['rmse']['count'] / G_rmse_mean_std['rmse']['size']
                subplot(axs[0], G_rmse_mean_std, 'G', axs[1], Dsm[0][0], Dsm[0][1])
                axs[1].set_axis_off()

            # Show the influence of different settings for the Discriminator
            else:  # nGsm == 1
                D_group = d_mr_mm_s_group.drop(gs + gm, axis='columns')
                D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['count', 'size'])
                D_rmse_mean_std['success_rate'] = D_rmse_mean_std['rmse']['count'] / D_rmse_mean_std['rmse']['size']
                subplot(axs[0], D_rmse_mean_std, 'D', axs[1], Gsm[0][0], Gsm[0][1])
                axs[1].set_axis_off()

            # Plot system information
            plot_info(axs[1], text, x=0.5)

        else:  # Multiple settings used for both the Generator and Discriminator
            # Plot parameters
            width, height, left, right, top, bottom, wspace, hspace = get_sizing(2, nrows, 12.8, 4.8)
            fig, axs = plt.subplots(nrows, 2, figsize=(width, height))
            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

            # Remove unused subplots and use the first and second empty subplot for the legend and system information
            info_ax, legend_ax, axs_set_axis_off, legend_loc = None, None, [], 'upper right'
            for i in range(nGsm):
                if i >= nDsm:
                    axs[i + 1, 0].set_axis_off()
                    if not info_ax: info_ax, legend_ax = axs[i + 1, 0], axs[i + 1, 0]

            for i in range(nDsm):
                if i >= nGsm:
                    axs[i + 1, 1].set_axis_off()
                    if not info_ax: info_ax, legend_ax = axs[i + 1, 1], axs[i + 1, 1]

            if not info_ax:
                width, height, left, right, top, bottom, wspace, hspace = get_sizing(3, nrows, 12.8, 4.8)
                fig, axs = plt.subplots(nrows, 3, figsize=(width, height))
                plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

                for i in range(nrows): axs[i, 2].set_axis_off()
                if not info_ax:
                    info_ax, legend_ax = axs[0, 2], axs[1, 2]
                    legend_loc = 'upper left'

            # Show the influence of different settings for the Generator (ignore Discriminator settings)
            G_group = d_mr_mm_s_group.drop(ds + dm, axis='columns')
            G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['count', 'size'])
            G_rmse_mean_std['success_rate'] = G_rmse_mean_std['rmse']['count'] / G_rmse_mean_std['rmse']['size']
            subplot(axs[0, 0], G_rmse_mean_std, 'G', legend_ax, legend_loc=legend_loc)

            # Show the influence of different settings for the Discriminator (ignore Generator settings)
            D_group = d_mr_mm_s_group.drop(gs + gm, axis='columns')
            D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['count', 'size'])
            D_rmse_mean_std['success_rate'] = D_rmse_mean_std['rmse']['count'] / D_rmse_mean_std['rmse']['size']
            subplot(axs[0, 1], D_rmse_mean_std, 'D', legend_ax)

            # Show the influence of different settings for the Generator for different Discriminator settings
            for i in range(nDsm):
                G_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['discriminator_sparsity'] == Dsm[i][0])
                    & (d_mr_mm_s_group['discriminator_modality'] == Dsm[i][1])
                )
                G_group.drop(ds + dm, axis='columns', inplace=True)
                G_rmse_mean_std = G_group.groupby(bs_hr_a_i + gs + gm, as_index=False).agg(['count', 'size'])
                G_rmse_mean_std['success_rate'] = G_rmse_mean_std['rmse']['count'] / G_rmse_mean_std['rmse']['size']
                subplot(axs[i + 1, 0], G_rmse_mean_std, 'G', legend_ax, Dsm[i][0], Dsm[i][1])

            # Show the influence of different settings for the Discriminator for different Generator settings
            for i in range(nGsm):
                D_group = d_mr_mm_s_group.where(
                    (d_mr_mm_s_group['generator_sparsity'] == Gsm[i][0])
                    & (d_mr_mm_s_group['generator_modality'] == Gsm[i][1])
                )
                D_group.drop(gs + gm, axis='columns', inplace=True)
                D_rmse_mean_std = D_group.groupby(bs_hr_a_i + ds + dm, as_index=False).agg(['count', 'size'])
                D_rmse_mean_std['success_rate'] = D_rmse_mean_std['rmse']['count'] / D_rmse_mean_std['rmse']['size']
                subplot(axs[i + 1, 1], D_rmse_mean_std, 'D', legend_ax, Gsm[i][0], Gsm[i][1])

            # Plot system information
            plot_info(info_ax, text)

        # Plot parameters
        plt.suptitle(title, size=24)

        if save:
            if verbose: print(f'Saving plot...')
            path = f'{folder}/{title}_success_rate.pdf'
            plt.savefig(path, format='pdf', dpi=1200)

        fig.show()
