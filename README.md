# Codebase for "Sparse Generative Adversarial Imputation Networks (SparseGAIN)"

Authors: Isil Baysal Erez, Brian van Oers

Paper: *reference to paper here*

Paper link: *link to paper here*

Contact: *contact information here*

This code is based on Jinsung Yoon, James Jordon and Mihaela van der Schaar's work ["GAIN: Missing Data Imputation using Generative Adversarial Nets," International Conference on Machine Learning (ICML), 2018.](https://github.com/jsyoon0823/GAIN) and adapted to use sparse initializations.
This is done with intent to increase performance: i.e. decrease the RMSE, decrease the memory requirement, decrease the run time, decrease the total FLOPs, decrease the failure rate, or any combination thereof, as compared to the original dense model and other imputation methods.

### Adaptations

- We fixed the seed of the xavier inititialization and the samplers to ensure consistency between results.
- We added a random sparse initialization method with a fixed seed based on the index of the passed layers.
- We added Erdos Renyi initialization for sparsity with a fixed seed based on the index of the passed layers.
- We added a loop to run main.py for different settings until the desired number of results for each setting is achieved.
- We added the functionality to save the imputations, the initialized weights of the generator and the success and failure counts.
- We added a jupyter notebook to compile the results and return the RMSE mean and standard deviation, the failure rate and total FLOPs, as well as plotting these in a graph.

### Possible improvements

- Change loop_main.py to a shell script to ensure tensorflow restarts every run. We have a problem with tensorflow slowing down each subsequent run and eventually running out of memory.
- Add more initializations: i.e. Erdos Renyi Kernel, SNIP and RSensitivity.
- Add more and larger datasets: i.e. CIFAR10, Fashion MNIST.
- Use Dynamic Sparse Training instead of Static Sparse Training.
- Implement the use of sparse tensors.

### Command inputs

- data_name: letter or spam
- miss_rate: probability of missing components
- batch_size: batch size
- hint_rate: hint rate
- alpha: hyperparameter
- iterations: iterations
- method: gain, iterative_imputer, iterative_imputer_rf
- init: xavier (dense, full), random, erdos_renyi (ER) (GAIN only)
- sparsity: generator sparsity level (Sparse GAIN only)
- save: save imputation result to a csv file
- folder: name of folder to save imputation results to

### Example command

```shell
$ python main.py --data_name spam --miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000 --method gain --init random --sparsity 0.9 --save --folder imputed_data
```

### Outputs

-   imputed_data_x: imputed data
-   rmse: Root Mean Squared Error

### References

*proper reference to the original GAIN paper here*

