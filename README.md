# Transfer Learning for ECG classification

This repository is a fork of [UTU-Health-Research/dl-ecg-classifier](https://github.com/UTU-Health-Research/dl-ecg-classifier). This adds fine-tuning code to the deep learning for ECG repository to support the experiments performed in the paper Patiño et al. (2026), "Transfer Learning for ECG Classification: Effects of Shared Labels, Granularity, and Fine-Tuning in Small Clinical Datasets", which was presented at the 39th IEEE International Symposium on Computer-Based Medical Systems (CBMS 2026).

This repository provides code to:

1. Pretrain ECG encoders on a source dataset/task.
2. Transfer learned representations to downstream ECG classification/regression tasks.
3. Evaluate models with standardized metrics and experiment tracking.
  
# Usage

To get started, install the required Python packages from the `requirements.txt` file using the following command:

```
pip install -r requirements.txt
```

The repository is tested with the Python version 3.13.1.

# Recommended folder repository structure
```
.
├── create_data_csvs.py
├── create_yaml_files.py
├── train_model.py
├── finetune_model.py
├── run_model.py
├── label_mapping.py
├── utils.py
├── data/
│   ├── preprocessed_data/                  # ECG files grouped by database
│   ├── split_csvs/
│   │   └── stratified_data/                # generated split CSVs
│   └── AHA_SNOMED_mapping.csv              # mapping file for SPH conversion
├── configs/                                # YAML configuration files
│   ├── training/
│   ├── predicting/
│   └── finetuning/
├── experiments/                            # training/prediction outputs
├── model_source/
│   └── source_multi/                       # source checkpoints for transfer learning
│   └── source_bin/
│   └── source_cat/
└── src/
    ├── modeling/
    │   ├── train_utils.py
    │   ├── finetune_utils.py
    │   ├── predict_utils.py
    │   ├── metrics.py
    │   └── models/
    │       └── seresnet18.py
    └── dataloader/
        └── dataset.py
        └── dataset_utils.py
        └── transforms.py                      
```

# Fine-tuning (Transfer Learning)

Script: **finetune_model.py**

This script fine-tunes from a pretrained (source) model checkpoint stored under:

```
model_source/<source_model>/<args.model>
```
Example:

* model_source/source_9_multi/split_1_1.pth

To run fine-tuning, the following code can be run in terminal. The first argument can either be a folder or a single YAML file. The second argument is the directory name of the source model.
```
python finetune_model.py <finetuning_yaml_or_dir> <source_model_dir_name>
```

# Citation

```
@article{patino2026transferlearning,
  title={Transfer Learning for ECG Classification: Effects of Shared Labels, Granularity, and Fine-Tuning in Small Clinical Datasets},
  author={Patiño, Chito and Kaisti, Matti and Pahikkala, Tapio and Airola, Antti},
  journal={Proceedings of the 39th IEEE International Symposium on Computer-Based Medical Systems},
  year={2026},
  address={Limassol, Cyprus},
  year={2026},
  publisher={IEEE},
  month={June}
}
```
