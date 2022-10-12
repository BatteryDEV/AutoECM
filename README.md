# Machine Learning-based Classification of Electrochemical Impedance Spectra

This folder contains the code to the corresponding publication:

"MACHINE LEARNING BENCHMARKS FOR THE CLASSIFICATION AND PARAMETERIZATION OF EQUIVALENT CIRCUIT MODELS FROM SOLID STATE ELECTROCHEMICAL IMPEDANCE SPECTRA"
Available on the archiv.


## Setup

A `requirements.txt` file and, alternatively, a file `environment.yml` is provided to create the `python` environment needed to run this code.

Firstly to set up the environment, please run the following:

```bash
conda env create --name eis-ml
conda activate eis-ml
pip install -r requirements.txt
```

## Workflow 

1. Run the `preprocess.py` file to calculate all required data, features, and images. (The GitHub repository comes with the data for the RF and CNN models. The tsfresh feature files are too large; therefore, this step is required if you want to run the xgb model.)
2. Run the model of your choice. The results are automatically saved in the respective results folder. The name of the folder is based on the timestamp.

Notebooks: 
`data_vis_and_exploration.ipynb`: Exploring the data and making plots.

.py files: 
`utils.py` Code to make plots
`utils_preprocessing.py` Code to preprocess the .csv files with the spectra

`preprocess.py` Preprocessing of data

`clf_rf.py`  Random forest model python script (train, test, save results)
`clf_xgb.py` XGB model python script (train, test, save results)
`clf_cnn.py` CNN model python script (train, test, save results)

eis folder: 
EIS toolkit that Raymond Gasper wrote for the BatteryDEV competition. 
Includes advanced options to simulate EIS data, visualize EIS data, and optimize equivalent circuit parameters based on initial guesses.

miscellaneous folder: 
Contains code to reproduce results shown in the supplementary information


## Contribute

We welcome any further contributions to this repository. If you find bucks or are interested in new features and capabilities, please raise an issue or propose a pull request.


## Data

This repository contains EIS spectra provided by QuantumScape, capturing about ten years of R&D.
The dataset consists of labeled data (~10k spectra). This means spectra with an associated Equivalent Circuit Model (ECM) with parameters estimated by Quantum Scape engineers.
Furthermore, the repository contains unlabeled spectra (~19k spectra). We thank Tim Holme from Quantum Scape for providing this dataset.

## License

The codecode in this repository is made publicly available under the terms of the MIT license as denoted in the `LICENSE` file. 
The data in this article is shared under the terms of the CC-BY 4.0 license according to the file `data\LICENSE`.

## Acknowledment/Citation

If you use code from this repositroy for your work, please cite:
