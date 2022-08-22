# Battmen

This folder contains the Code to the corresponsing publication:

The two main appriaches were developed by the teams: 
Battmen: Paul Gasper, Rhys Goodall, Tushar Desai, Andrew Schiek, Hugo Leduc
TeamCNN:


Here's Joachim, I restructured the code a bit and fixed a few bugs.
To use a new dataset, the pipline params must be reset:
https://github.com/blue-yonder/tsfresh/blob/main/notebooks/examples/02%20sklearn%20Pipeline.ipynb
This was previosuly not done and let to weird results on the held out test dataset.

Notebooks: 

classification.ipynb: Training testing the model with different splits. The results on the hold out test datset look almost too good. @Ray can you check them?
data_exploration_plots: Exploring the data and making plots.
param_regr: Looks like leftovers from experimentation. Can probably be deleted. 

py files: 
utils.py: Code to make plots
eis_preprocessing.py: Code to preprocess the .csv files with the spectra

eis_tsfresh.xgb.py: Call this script to train or pred. 
USAGE:
```bash
python eis_tsfresh_xgb.py -h
usage: eis_tsfresh_xgb.py [-h] [--mode MODE] [--modelpath MODELPATH]
                          [--datapath DATAPATH] [--pred_data PRED_DATA]

EIS spectra processing with ML

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Only 'train_clf', 'train_reg' or 'pred' are valid
                        inputs. You need to train the model beofre calling
                        prediction if you pass no argument, the cleassifier
                        and the predictor will be trained and predicitons will
                        be made
  --modelpath MODELPATH
                        Path where to store/load the model from
  --datapath DATAPATH   Path where to load the data from. In this directory,
                        there must be a file called train_data.csv adn
                        test_data.csv
  --pred_data PRED_DATA
                        .csv file to use for making predictions. If no
                        argument is supplied, the script will look for
                        test_data.csv under datapath
```


OLD ReadME: 
## Setup

A `submit-environment.yml` file is provided to create the conda environment needed to run this code. Although we did not make use of `dvc`, instead prototyping on google colab, we have provided `dvc` style YAML files to show how the scripts should be run.

Firstly to setup the environment please run:

```bash
conda env create --file environment.yml
conda activate eis-battmen
```

## Summary of work

Initially we considered making use of a Neural Controlled Differential Equation (https://github.com/patrick-kidger/torchcde) model to model the EIS. However these models are expensive to train and on further thought the frequency does not have the causal structure of a time series and there are potentially issues with data leakage due to the fact that different frequency regimes are used predominately for different types of ECMs due to the habits of different QuantumScape researchers over time.

In order to avoid this key issue of data leakage we opted to use an interpolated basis for examining the EIS. The largest frequency range possible for which no extrapolation is required is between 10Hz and 100KHz. Interpolating onto this basis gives us a tabular data set. XGBoost is a gradient boosted modelling approach known to excel on tabular data sets, particularly of intermediate sizes such as considered in this challenge.

Simply fitting XGBoost to the raw tabular data gave a classification accuracy of approximately 40%. We made use of the `tsfresh` package to carry out additional feature extraction from the interpolated spectra. Using these additional features led to an accuracy of approximately 50%. The accuracy for distinguishing the distinct physical circuits (Rs_Ws, RC-G-G) was observed to be much higher in the range 85-95%. Whilst the accuracy for the more physically similar circuits was generally lower in the range 25-70%. This makes sense as the mapping of spectra to ECMs is somewhat qualitative as many different ECMs can provide reasonable fits to the same data. We explore the H2O autoML package and concluded that the additional complexity did not warrant the negligible improvement in performance observed.

Due to our inability to devise a end-to-end differentiable data structure for ECMs we were only able to tackle the regression task by making use of individual models for each class. This is not an ideal solution given the large similarities that exist between some of the tasks that may have been beneficial for learning if they could be incorporated in a holistic regression framework. Given earlier explorations we opted to use XGBoost again for this stage. When conducting an EDA on the parameters data we found that nearly all the parameters had heavy tailed distributions apart from the exponent parameter in constant phase element components which was approximately uniform. Consequently the most important aspect of the pipeline to tune was the scaling of the target labels. We explored the use of a log transform, the yeo-johnson transform and other standardisation approached for heavy tailed data. In the end the approach that worked best was a quantile transform. The regression model could perhaps be improved further if more time was available to tune the hyperparameters such as the number quantile bins. Because the magnitude of the errors for any given parameters may be extremely large, model performance was compared to a median dummy regressor.  Finding a model with performance better than the dummy regressor took several tries.

We have incorporated all the model training into a single python script and then prepared a second script that loads the serialised models to perform inference. Collab files used to develop the models are provided as Google Collaboratory links as well.

## Explored but not submitted

There were three main areas of work proposed by the team that we were not able to complete.

1. Generation of additional synthetic data: The QuantumScape data is a unknown mixture of experimental and synthetic data to the team's knowledge. Given the difficultly of distinguishing between similar ECMs the team wondered what the upper limit to performance might be. The best way to explore this is to generate idealised clean EIS for simulated circuits in large amounts and use this to train a model. The performance of a model trained on huge amounts of perfectly clean simulated data would give us an effective upper bound on the performance any model trained on noisy and imperfect experimental data might achieve. The availability of a simulated EIS dataset may also be of use in the training of effective neural network based classification models via the use of transfer learning.

2. Image-based classification of spectra: When initial efforts into using a Neural Controlled Differential Equation failed we explored the idea of using a computer vision model trained on normalized images of spectra. Careful work was required when deciding on how to prepare images of the spectra for this work (line thickness, resolution, file format). This effort highlighted that several dubious looking spectra exist in the data set as systems with discontinuities are easy to identify visually. However, whilst we began training a ResNet50 model on this task we ran out of time to properly train and validate this model.

3. In order to make significant progress for truly automated ECM labelling we need to develop a high throughput end-to-end system. The key aspects for this system are a differentiable representation of ECMs (i.e. a generative graph model) and automatic differentiation of impedance simulations. The challenge with automatic differentiation of complex numbers is that most automatic differentiation code bases are designed to handle the differentiation of real numbers and are only setup for certain type of complex differentiation (e.g. C -> R). A differentiable representation of ECMs is possible based on extensive literature for differentiable representations of molecular graphs however a key challenge is that in optimizing such a representation we need to ensure that we output parsimonious (i.e. simple) models. As such it would ned necessary to explore complexity measures for graphs and work out how to regularize/penalise the loss in order to ensure that simple ECMs were favoured.

## Summary

We have enjoyed working on the QuantumScape challenge and hope that the judges appreciate the efforts of the team throughout the hackathon.

Battmen.

olab Links: (maybe we should nort encourage people do use colab in that way_). 
It would be better to create a new colab and cloen the github if people wnat to use colab for development.
Classification: 
https://colab.research.google.com/drive/10EGG_JnUIc_zvifIQiaXUZojM8iD-udN?usp=sharing

Regression:
https://colab.research.google.com/drive/1zCLW5xoKo7xu4VSe-YeabJ1SsIW90OFV?usp=sharing
