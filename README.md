## Comprehensive Evaluation of Vulnerability Detection Models towards Making Perfect Predictions

VulGraphPTM is a model that leverages the global insights of PTM and the local information of GNN to bolster the model’s expressive power over vulnerability semantics, thus aims to better differentiate vulnerable functions from their patched versions.

### Setup
In the `/VulGraphPTM` path, there is a `requirements.txt` file. Run the following command to install the necessary Python environment.
```
$ pip install -r requirements.txt
```

### Quick Start
We provide pre-trained model checkpoints and processed VulFixed vulnerability-patch pairs dataset to facilitate users to quickly get started with our work.

#### To get the model ready
Enter the `/VulGraphPTM/models/VulFixed` directory, run:
```
$ bash download.sh
```

#### To get the processed data ready
Enter the `/processed_data` directory, run:
```
$ bash download.sh
$ tar -zxvf Balance-map_full_data_processed.tar.gz
```

Then we can prepare to start running our work.
#### Run test
Enter the `/VulGraphPTM` directory, locate the main.py script, and run:
```
$ python main.py --model_type bertgam --action test --dataset Balance-map_test --input_dir ../processed_data/Balance-map_full_data_processed --feature_size 169 --calc_diff
```
The `--calc_diff` parameter requires the model to calculate results such as `Pair_acc` for the predicted paired instances.
Then you will get results such as:
```
####################################################################################################
All vuln: 29.79 All safe: 25.85 Reversed: 6.13  Correct: 38.23
models/VulFixed/BertwithGGAP    Test Accuracy: 66.09    Precision: 66.05        Recall: 68.02   F1: 67.02
====================================================================================================
```

#### Retrain
Run the following command to retrain the model:
```
python main.py --model_type bertgam --action train --dataset Balance-map --input_dir ../processed_data/Balance-map_full_data_processed --feature_size 169
```

### Other datasets
If you want to run other datasets, we provide an `encoder.py` script under the `/VulGraphPTM` directory. You just need to modify the dataset path in the main function of the script to run and generate custom processed data.
