# Machine Learning-based Classification of Electrochemical Impedance Spectra

### Clarifications regarding the distribution of synthetic and measured impedance spectra in the labelled and unlabeled portion of the data set have been made. 26th of April 2023

This repository contains the code to the corresponding publication:
"Machine learning benchmarks for the classification of equivalent circuit models from solid-state electrochemical impedance spectra"
http://arxiv.org/abs/2302.03362

## Setup

A `requirements.txt` file and, alternatively, a file `environment.yml` is provided to create the `python` environment needed to run this code.

Firstly to set up the environment, please run the following:

```bash
conda create -n "eis-ml" python=3.9.15 ipython
conda activate eis-ml
pip install -r requirements.txt
```
If you are on MacOS you might have to install TensorFlow and tsfresh separately/manually fix the installation of sub-dependencies etc.

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

QuantumScape (QS) provided the EIS data contained in the repository. The first data set comprises approximately 9,300 synthetic spectra with the associated Equivalent Circuit Model (ECM). 
The second data set contains approximately 19,000 unlabeled spectra consisting of about 80% synthetic and 20% measured data. 
The parameter ranges for all synthetic data are informed by the R&D of QS. The measured spectra are from a range of different materials, with some replicate measurements at different temperatures, and/or State-Of-Charge (SOC), and/or State-Of-Health (SOH).

The labeled data set is split into training and test data set: `data/train_data.csv` and `data/test_data.csv`
Furthermore, the repository contains unlabeled spectra (~19k spectra): `data/unlabeled_data.csv`. 
The data in this article is shared under the terms of the CC-BY 4.0 license according to the file `data\LICENSE`.
We thank Tim Holme from Quantum Scape for providing these data sets.

## License

The code in this repository is made publicly available under the terms of the MIT license as denoted in the `LICENSE` file. 
The data in this article is shared under the terms of the CC-BY 4.0 license according to the file `data\LICENSE`.

## Acknowledgment/Citation

If you use code from this repository for your work, please cite: 


@article{Schaeffer_2023,  
	author = {Joachim Schaeffer and Paul Gasper and Esteban Garcia-Tamayo and Raymond Gasper and Masaki Adachi and Juan Pablo Gaviria-Cardona and Simon Montoya-Bedoya and Anoushka Bhutani and Andrew Schiek and Rhys Goodall and Rolf Findeisen and Richard D. Braatz and Simon Engelke},  
 	doi = {[10.1149/1945-7111/acd8fb](10.1149/1945-7111/acd8fb)},  
	journal = {Journal of The Electrochemical Society},  
	month = {jun},  
	number = {6},  
	pages = {060512},  
	publisher = {IOP Publishing},  
	title = {Machine Learning Benchmarks for the Classification of Equivalent Circuit Models from Electrochemical Impedance Spectra},  
	volume = {170},  
	year = {2023},  
}  
