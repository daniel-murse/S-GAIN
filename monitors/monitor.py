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

"""Monitor class for S-GAIN:

Todo: run in separate thread

(1) init_monitors: initialize the temporary folder
(2) start_imputation_time_monitor: open the imputation time log file
(3) start_rmse_monitor: open the RMSE log file
(4) start_memory_usage_monitor: open the memory usage log file
(5) start_energy_consumption_monitor: open the energy consumption log file
(6) start_sparsity_monitor: open the sparsity log files
(7) start_flops_monitor: open the FLOPs log files
(8) start_loss_monitor: open the loss log files
(9) start_all_monitors: start all the monitors
(10) log_rmse: log the RMSE
(11) log_imputation_time: log the imputation time
(12) log_memory_usage: log the memory usage
(13) log_energy_consumption: log the energy consumption
(14) log_sparsity: log the sparsity
(15) log_flops: log the FLOPs
(16) log_loss: log the loss
(17) log_all_monitors: log the all monitors
(18) stop_rmse_monitor: close the RMSE log file
(19) stop_imputation_time_monitor: close the imputation time log file
(20) stop_memory_usage_monitor: close the memory usage log file
(21) stop_energy_consumption_monitor: close the energy consumption log file
(22) stop_sparsity_monitor: close the sparsity log files
(23) stop_flops_monitor: close the FLOPs log files
(24) stop_loss_monitor: close the loss log files
(25) stop_all_monitors: close all the monitors
(26) save_logs: compile and save the logs to a json file
(27) set_model: set the (trained) model, so it can be saved later
(28) save_model: save the (trained) model to a json file
"""

import json

from os import makedirs
from os.path import isdir
from shutil import rmtree
from time import time

from utils.load_store import parse_experiment, read_bin
from utils.metrics import get_rmse


class Monitor:
    def __init__(self, data_x, data_mask, experiment=None, directory='temp', verbose=False):
        """Initialize the monitor.

        :param data_x: the original data (without missing values)
        :param data_mask: the indicator matrix for missing elements
        :param experiment: the name of the experiment (optional)
        :param directory: the temporary directory
        :param verbose: enable verbose output to console
        """

        # Parameters
        self.data_x = data_x
        self.data_mask = data_mask
        self.experiment = experiment
        self.directory = directory
        self.verbose = verbose

        # Log variables
        self.imputation_time = None

        # Log files
        self.f_RMSE = None
        self.f_imputation_time = None
        self.f_memory_usage = None
        self.f_energy_consumption = None
        self.f_sparsity_G, self.f_sparsity_G_W1, self.f_sparsity_G_W2, self.f_sparsity_G_W3 = [None] * 4
        self.f_sparsity_D, self.f_sparsity_D_W1, self.f_sparsity_D_W2, self.f_sparsity_D_W3 = [None] * 4
        self.f_FLOPs_G, self.f_FLOPs_D = [None] * 2
        self.f_loss_G, self.f_loss_D, self.f_loss_MSE = [None] * 3

        # Model
        self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3 = [None] * 6
        self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3 = [None] * 6

    def init_monitor(self):
        """Initialize the temporary folder."""

        # Clear the logs and create temporary directory
        if self.verbose: print('Initializing monitor...')
        if isdir(self.directory): rmtree(self.directory)
        makedirs(self.directory)

    # Start monitors
    def start_rmse_monitor(self):
        """Open the RMSE log file and start monitoring.

        :return: True
        """

        self.f_RMSE = open(f'{self.directory}/rmse.bin', 'ab')

        # if self.verbose: print('Monitoring RMSE...')
        return True

    def start_imputation_time_monitor(self):
        """Open the imputation time log file and start monitoring.

        :return: True
        """

        self.f_imputation_time = open(f'{self.directory}/imputation_time.bin', 'ab')
        self.imputation_time = time()

        if self.verbose: print('Monitoring imputation time...')
        return True

    def start_memory_usage_monitor(self):
        """Open the memory usage log file and start monitoring.

        :return: True
        """

        self.f_memory_usage = open(f'{self.directory}/memory_usage.bin', 'ab')

        # if self.verbose: print('Monitoring memory usage...')
        return True

    def start_energy_consumption_monitor(self):
        """Open the energy consumption log file and start monitoring.

        :return: True
        """

        self.f_energy_consumption = open(f'{self.directory}/energy_consumption.bin', 'ab')

        # if self.verbose: print('Monitoring energy consumption...')
        return True

    def start_sparsity_monitor(self):
        """Open the sparsity log files and start monitoring.

        :return: True
        """

        self.f_sparsity_G = open(f'{self.directory}/sparsity_G.bin', 'ab')
        self.f_sparsity_G_W1 = open(f'{self.directory}/sparsity_G_W1.bin', 'ab')
        self.f_sparsity_G_W2 = open(f'{self.directory}/sparsity_G_W2.bin', 'ab')
        self.f_sparsity_G_W3 = open(f'{self.directory}/sparsity_G_W3.bin', 'ab')

        self.f_sparsity_D = open(f'{self.directory}/sparsity_D.bin', 'ab')
        self.f_sparsity_D_W1 = open(f'{self.directory}/sparsity_D_W1.bin', 'ab')
        self.f_sparsity_D_W2 = open(f'{self.directory}/sparsity_D_W2.bin', 'ab')
        self.f_sparsity_D_W3 = open(f'{self.directory}/sparsity_D_W3.bin', 'ab')

        # if self.verbose: print('Monitoring sparsity...')
        return True

    def start_flops_monitor(self):
        """Open the FLOPs log files and start monitoring.

        :return: True
        """

        self.f_FLOPs_G = open(f'{self.directory}/flops_G.bin', 'ab')
        self.f_FLOPs_D = open(f'{self.directory}/flops_D.bin', 'ab')

        # if self.verbose: print('Monitoring FLOPs...')
        return True

    def start_loss_monitor(self):
        """Open the loss log files and start monitoring.

        :return: True
        """

        self.f_loss_G = open(f'{self.directory}/loss_G.bin', 'ab')
        self.f_loss_D = open(f'{self.directory}/loss_D.bin', 'ab')
        self.f_loss_MSE = open(f'{self.directory}/loss_MSE.bin', 'ab')

        if self.verbose: print('Monitoring loss...')
        return True

    def start_all_monitors(self):
        """Start all the monitors.

        :return: True
        """

        if self.verbose: print('Starting monitors...')
        self.start_imputation_time_monitor()
        self.start_memory_usage_monitor()
        self.start_energy_consumption_monitor()
        self.start_sparsity_monitor()
        self.start_flops_monitor()
        self.start_loss_monitor()
        self.start_rmse_monitor()

        return True

    # Log metrics
    def log_rmse(self, imputed_data):
        """Log the RMSE.

        :param imputed_data: The imputed data

        :return:
        - RMSE: the Root Mean Square Error
        """

        RMSE = get_rmse(self.data_x, imputed_data, self.data_mask)
        self.f_RMSE.write(struct.pack('f', RMSE))

        return RMSE

    def log_imputation_time(self):
        """Log the imputation time.

        :return:
        - step_time: the time (in seconds) between the previous step and now
        """

        current_time = time()
        step_time = current_time - self.imputation_time
        self.f_imputation_time.write(struct.pack('f', step_time))
        self.imputation_time = current_time

        return step_time

    def log_memory_usage(self):
        """Log the memory usage.

        :return: True
        """

        # Todo
        self.f_memory_usage.write()

        return True

    def log_energy_consumption(self):
        """Log the energy consumption.

        :return: True
        """

        # Todo
        self.f_energy_consumption.write()

        return True

    def log_sparsity(self, G_W1_sparsity, G_W2_sparsity, G_W3_sparsity, D_W1_sparsity, D_W2_sparsity, D_W3_sparsity):
        """Log the sparsity.

        :param G_W1_sparsity: the sparsity of the first layer of the Generator
        :param G_W2_sparsity: the sparsity of the second layer of the Generator
        :param G_W3_sparsity: the sparsity of the third layer of the Generator
        :param D_W1_sparsity: the sparsity of the first layer of the Discriminator
        :param D_W2_sparsity: the sparsity of the second layer of the Discriminator
        :param D_W3_sparsity: the sparsity of the third layer of the Discriminator

        :return: True
        """

        # Todo

        self.f_sparsity_G.write()
        self.f_sparsity_G_W1.write()
        self.f_sparsity_G_W2.write()
        self.f_sparsity_G_W3.write()

        self.f_sparsity_D.write()
        self.f_sparsity_D_W1.write()
        self.f_sparsity_D_W2.write()
        self.f_sparsity_D_W3.write()

        return True

    def log_flops(self):
        """Log the FLOPs.

        :return: True
        """

        # Todo
        self.f_FLOPs_G.write()
        self.f_FLOPs_D.write()

        return True

    def log_loss(self, loss_G, loss_D, loss_MSE):
        """Log the loss.

        :param loss_G: the loss of the generator (cross entropy)
        :param loss_D: the loss of the discriminator (cross entropy)
        :param loss_MSE: the loss (MSE)

        :return: True
        """

        self.f_loss_G.write(struct.pack('f', loss_G))
        self.f_loss_D.write(struct.pack('f', loss_D))
        self.f_loss_MSE.write(struct.pack('f', loss_MSE))

        return True

    def log_all(self, imputed_data, G_W1_sparsity, G_W2_sparsity, G_W3_sparsity, D_W1_sparsity, D_W2_sparsity,
                D_W3_sparsity, loss_G, loss_D, loss_MSE):
        """Log the all monitors.

        :param imputed_data: The imputed data
        :param G_W1_sparsity: the sparsity of the first layer of the Generator
        :param G_W2_sparsity: the sparsity of the second layer of the Generator
        :param G_W3_sparsity: the sparsity of the third layer of the Generator
        :param D_W1_sparsity: the sparsity of the first layer of the Discriminator
        :param D_W2_sparsity: the sparsity of the second layer of the Discriminator
        :param D_W3_sparsity: the sparsity of the third layer of the Discriminator
        :param loss_G: the loss of the generator (cross entropy)
        :param loss_D: the loss of the discriminator (cross entropy)
        :param loss_MSE: the loss (MSE)

        :return: True
        """

        # Todo
        self.log_rmse(imputed_data)
        self.log_imputation_time()
        self.log_memory_usage()
        self.log_energy_consumption()
        self.log_sparsity(G_W1_sparsity, G_W2_sparsity, G_W3_sparsity, D_W1_sparsity, D_W2_sparsity, D_W3_sparsity)
        self.log_flops()
        self.log_loss(loss_G, loss_D, loss_MSE)

        return True

    # Stop monitors
    def stop_rmse_monitor(self):
        """Close the RMSE log file and stop monitoring.

        :return: False
        """

        self.f_RMSE.close()

        if self.verbose: print('Stopped monitoring RMSE.')
        return False

    def stop_imputation_time_monitor(self):
        """Close the imputation time log file and stop monitoring.

        :return: False
        """

        self.f_imputation_time.close()

        if self.verbose: print('Stopped monitoring imputation time.')
        return False

    def stop_memory_usage_monitor(self):
        """Close the memory usage log file and stop monitoring.

        :return: False
        """

        self.f_memory_usage.close()

        # if self.verbose: print('Stopped monitoring memory usage.')
        return False

    def stop_energy_consumption_monitor(self):
        """Close the energy consumption log file and stop monitoring.

        :return: False
        """

        self.f_energy_consumption.close()

        # if self.verbose: print('Stopped monitoring energy consumption.')
        return False

    def stop_sparsity_monitor(self):
        """Close the sparsity log files and stop monitoring.

        :return: False
        """

        self.f_sparsity_G.close()
        self.f_sparsity_G_W1.close()
        self.f_sparsity_G_W2.close()
        self.f_sparsity_G_W3.close()

        self.f_sparsity_D.close()
        self.f_sparsity_D_W1.close()
        self.f_sparsity_D_W2.close()
        self.f_sparsity_D_W3.close()

        # if self.verbose: print('Stopped monitoring sparsity.')
        return False

    def stop_flops_monitor(self):
        """Close the FLOPs log files and stop monitoring.

        :return: False
        """

        self.f_FLOPs_G.close()
        self.f_FLOPs_D.close()

        # if self.verbose: print('Stopped monitoring FLOPs.')
        return False

    def stop_loss_monitor(self):
        """Close the loss log files and stop monitoring.

        :return: False
        """

        self.f_loss_G.close()
        self.f_loss_D.close()
        self.f_loss_MSE.close()

        if self.verbose: print('Stopped monitoring loss (cross entropy and MSE).')
        return False

    def stop_all_monitors(self):
        """Stop all the monitors.

        :return: False
        """

        self.stop_rmse_monitor()
        self.stop_imputation_time_monitor()
        self.stop_memory_usage_monitor()
        self.stop_energy_consumption_monitor()
        self.stop_sparsity_monitor()
        self.stop_flops_monitor()

        if self.verbose: print('Stopped monitors.')
        return False

    # Compile and save the logs
    def save_logs(self, filepath):
        """Compile and save the logs to a json file.

        :param filepath: the filepath to save the logs to

        :return:
        - RMSE: the RMSE log
        - imputation_time: the imputation time log
        - memory_usage: the memory usage log
        - energy_consumption: the energy consumption log
        - sparsity: the sparsity log (total)
        - sparsity_G: the sparsity log for the generator
        - sparsity_G_W1: the sparsity log for the first layer of the generator
        - sparsity_G_W2: the sparsity log for the second layer of the generator
        - sparsity_G_W3: the sparsity log for the third layer of the generator
        - sparsity_D: the sparsity log for the discriminator
        - sparsity_D_W1: the sparsity log for the first layer of the discriminator
        - sparsity_D_W2: the sparsity log for the second layer of the discriminator
        - sparsity_D_W3: the sparsity log for the third layer of the discriminator
        - FLOPs: the FLOPs log (total)
        - FLOPs_G: the FLOPs log for the generator
        - FLOPs_D: the FLOPs log for the discriminator
        - loss_G: the loss log for the generator (cross entropy)
        - loss_D: the loss log for the discriminator (cross entropy)
        - loss_MSE: the loss log (MSE)
        """

        if self.verbose: print('Saving logs...')

        # Read the log files
        RMSE = read_bin(f'{self.directory}/RMSE.bin')
        imputation_time = read_bin(f'{self.directory}/imputation_time.bin')
        memory_usage = read_bin(f'{self.directory}/memory_usage.bin')
        energy_consumption = read_bin(f'{self.directory}/energy_consumption.bin')
        sparsity_G = read_bin(f'{self.directory}/sparsity_G.bin')
        sparsity_G_W1 = read_bin(f'{self.directory}/sparsity_G_W1.bin')
        sparsity_G_W2 = read_bin(f'{self.directory}/sparsity_G_W2.bin')
        sparsity_G_W3 = read_bin(f'{self.directory}/sparsity_G_W3.bin')
        sparsity_D = read_bin(f'{self.directory}/sparsity_D.bin')
        sparsity_D_W1 = read_bin(f'{self.directory}/sparsity_D_W1.bin')
        sparsity_D_W2 = read_bin(f'{self.directory}/sparsity_D_W2.bin')
        sparsity_D_W3 = read_bin(f'{self.directory}/sparsity_D_W3.bin')
        FLOPs_G = read_bin(f'{self.directory}/flops_G.bin')
        FLOPs_D = read_bin(f'{self.directory}/flops_D.bin')
        loss_G = read_bin(f'{self.directory}/loss_G.bin')
        loss_D = read_bin(f'{self.directory}/loss_D.bin')
        loss_MSE = read_bin(f'{self.directory}/loss_MSE.bin')

        # Todo totals
        sparsity = [None]
        FLOPs = [0]

        logs = {}
        if self.experiment is not None:
            dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity, \
                generator_modality, discriminator_sparsity, discriminator_modality \
                = parse_experiment(self.experiment, file=False)

            logs.update({
                'experiment': {
                    'dataset': dataset,
                    'miss_rate': miss_rate,
                    'miss_modality': miss_modality,
                    'seed': seed,
                    'batch_size': batch_size,
                    'hint_rate': hint_rate,
                    'alpha': alpha,
                    'iterations': iterations,
                    'generator_sparsity': generator_sparsity,
                    'generator_modality': generator_modality,
                    'discriminator_sparsity': discriminator_sparsity,
                    'discriminator_modality': discriminator_modality
                }
            })

        logs.update({
            'rmse': {
                'final': RMSE[-1],
                'log': RMSE,
            },
            'imputation_time': {
                'total': sum(imputation_time),
                'log': imputation_time
            },
            'memory_usage': {
                'maximum': max(memory_usage),
                'average': sum(memory_usage) / len(memory_usage),
                'log': memory_usage
            },
            'energy_consumption': {
                'total': sum(energy_consumption),
                'log': energy_consumption
            },
            'sparsity': {
                'initial': sparsity[0],
                'final': sparsity[-1],
                'log': sparsity,
                'generator': {
                    'initial': sparsity_G[0],
                    'final': sparsity_G[-1],
                    'log': sparsity_G,
                    'G_W1': {
                        'initial': sparsity_G_W1[0],
                        'final': sparsity_G_W1[-1],
                        'log': sparsity_G_W1
                    },
                    'G_W2': {
                        'initial': sparsity_G_W2[0],
                        'final': sparsity_G_W2[-1],
                        'log': sparsity_G_W2
                    },
                    'G_W3': {
                        'initial': sparsity_G_W3[0],
                        'final': sparsity_G_W3[-1],
                        'log': sparsity_G_W3
                    }
                },
                'discriminator': {
                    'initial': sparsity_D[0],
                    'final': sparsity_D[-1],
                    'log': sparsity_D,
                    'D_W1': {
                        'initial': sparsity_D_W1[0],
                        'final': sparsity_D_W1[-1],
                        'log': sparsity_D_W1
                    },
                    'D_W2': {
                        'initial': sparsity_D_W2[0],
                        'final': sparsity_D_W2[-1],
                        'log': sparsity_D_W2
                    },
                    'D_W3': {
                        'initial': sparsity_D_W3[0],
                        'final': sparsity_D_W3[-1],
                        'log': sparsity_D_W3
                    }
                }
            },
            'flops': {
                'total': sum(FLOPs),
                'log': FLOPs,
                'generator': {
                    'total': sum(FLOPs_G),
                    'log': FLOPs_G
                },
                'discriminator': {
                    'total': sum(FLOPs_D),
                    'log': FLOPs_D
                }
            },
            'loss': {
                'cross_entropy': {
                    'generator': {
                        'initial': loss_G[0],
                        'total': loss_G[-1],
                        'log': loss_G
                    },
                    'discriminator': {
                        'initial': loss_D[0],
                        'final': loss_D[-1],
                        'log': loss_D
                    }
                },
                'MSE': {
                    'initial': loss_MSE[0],
                    'final': loss_MSE[-1],
                    'log': loss_MSE
                }
            }
        })

        f_logs = open(filepath, 'w')
        f_logs.write(json.dumps(logs))
        f_logs.close()

        return RMSE, imputation_time, memory_usage, energy_consumption, sparsity, sparsity_G, sparsity_G_W1, \
            sparsity_G_W2, sparsity_G_W3, sparsity_D, sparsity_D_W1, sparsity_D_W2, sparsity_D_W3, FLOPs, FLOPs_G, \
            FLOPs_D, loss_G, loss_D, loss_MSE

    def set_model(self, theta_G, theta_D):
        """Set the (trained) model, so it can be saved later.

        :param theta_G: the generator variables: G_W1, G_W2, G_W3, G_b1, G_b2, G_b3
        :param theta_D: the discriminator variables: D_W1, D_W2, D_W3, D_b1, D_b2, D_b3
        """

        self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3 = theta_G
        self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3 = theta_D

    def save_model(self, filepath):
        """Save the (trained) model to a json file.

        :param filepath: the filepath to save the model to
        """

        model = json.dumps({
            'theta_G': {
                'G_W1': self.G_W1.tolist(),
                'G_W2': self.G_W2.tolist(),
                'G_W3': self.G_W3.tolist(),
                'G_b1': self.G_b1.tolist(),
                'G_b2': self.G_b2.tolist(),
                'G_b3': self.G_b3.tolist()
            },
            'theta_D': {
                'D_W1': self.D_W1.tolist(),
                'D_W2': self.D_W2.tolist(),
                'D_W3': self.D_W3.tolist(),
                'D_b1': self.D_b1.tolist(),
                'D_b2': self.D_b2.tolist(),
                'D_b3': self.D_b3.tolist()
            }
        })

        f_model = open(filepath, 'w')
        f_model.write(model)
        f_model.close()
