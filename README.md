# Machine Learning based Classification of Electrochemical Impedance Spectra

This folder contains the Code to the corresponsing publication:

"MACHINE LEARNING BENCHMARKS FOR THE CLASSIFICATION AND PARAMETERIZATION OF EQUIVALENT CIRCUIT MODELS FROM SOLID STATE ELECTROCHEMICAL IMPEDANCE SPECTRA"
Available on archiv.


## Setup

A `requirements.txt` file and alternatively a `environment.yml` file is provided to create the `python` environment needed to run this code.

Firstly to setup the environment please run:

```bash
conda env create --file requirements.txt
conda activate eis-battmen
```


## Workflow 

1. Run the `preprocess.py` file to calculate all required data, features and images. (The github repository comes with the data for the RF and CNN models. The xgboost+tsfresh files are too large and therefore this step is required if you want to run the xgb model.)
2. Run the model of your choice. The results are atuomatically saved in the respective results folder. The name of the folder is based in the timestamp.

Notebooks: 
`data_vis_and_exploration.ipynb`: Exploring the data and making plots.

py files: 
`utils.py` Code to make plots
`utils_preprocessing.py` Code to preprocess the .csv files with the spectra

`preprocess.pyz` Pre-processing of data

`clf_rf.py`  Random forest model python script (train, test, save results)
`clf_xgb.py` XGB model python script (train, test, save results)
`clf_cnn.py` CNN model python script (train, test, save results)

eis folder: 
eis toolkit written by Raymond Gasper for the BatteryDEV competition. 
Includes advaced options to simulate EIS data, visualize EIS data and optimized equivalent circuit parameters based on initial guesses.

miscellaneous folder: 
Contains code to reproduce results shown in the supplementatary information. 


## Contribute

We welcome any further contirbutions to this repository. Incase you find bucks or are intersted in new features and capabilities, please raise an issur or propose a pull request.