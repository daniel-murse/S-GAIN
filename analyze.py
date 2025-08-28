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


"""Analyze the experiments."""

import argparse

from os import listdir

from utils.load_store import parse_files
from utils.analysis import compile_metrics, plot_rmse, plot_success_rate


def main(args):
    """Plot the graphs.

    :param args:
    - all: plot all graphs
    - rmse: plot the RMSE
    - success_rate: plot the success rate
    - save: save the plots
    - input: the folder the experiments were saved to
    - output: the folder to save the analysis to
    """

    # Get the parameters
    plot_all = args.all
    rmse = args.rmse
    success_rate = args.success_rate
    save = args.save
    input_folder = args.input
    output_folder = args.output

    # Get all log files
    logs = [file for file in listdir(input_folder) if file.endswith('log.json')]

    # Load experiments
    experiments = parse_files(logs)

    # Analyze experiments
    compile_metrics(experiments, save=save, folder=output_folder)
    if plot_all or rmse: plot_rmse(experiments, save=save, folder=output_folder)
    if plot_all or success_rate: plot_success_rate(experiments, save=save, folder=output_folder)

    # Drop failed experiments
    logs = [file for file in logs if 'nan' not in file]

    # Continue analysis
    # Todo the rest of the analysis


if __name__ == '__main__':
    # Inputs for the analysis function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--all',
        help="plot all the graphs",
        action='store_true')
    parser.add_argument(
        '-rmse', '--rmse',
        help="plot the RMSE graphs",
        action='store_true')
    parser.add_argument(
        '-sr', '--success_rate',
        help="plot the success rate graphs",
        action='store_true')
    parser.add_argument(
        '-s', '--save',
        help="save the analysis",
        action='store_true')
    parser.add_argument(
        '-in', '--input', '--experiments',
        help='the folder the experiments were saved to (optional) [default: output]',
        default='output',
        type=str)
    parser.add_argument(
        '-out', '--output', '--analysis',
        help='save the analysis to a different folder (optional) [default: analysis]',
        default='analysis',
        type=str)
    args = parser.parse_args()

    main(args)