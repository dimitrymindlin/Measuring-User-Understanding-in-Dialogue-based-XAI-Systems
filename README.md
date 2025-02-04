# Backend - Measuring-User-Understanding-in-Dialogue-based-XAI-Systems

Repository for the backend of the paper "Measuring User Understanding in Dialogue based xAI Systems".
Currently working for the UCI adult census dataset as described in the paper.
Additionally,
the [experiment frontend](https://github.com/dimitrymindlin/Measuring-User-Understanding-in-Dialogue-based-XAI-Systems-Frontend.git)
is needed to run the experiment UI.

## Table of Contents
- [Installation](#installation)
- [Starting the Experiment locally](#starting-the-experiment-locally)
- [Starting the Experiment in docker](#starting-the-experiment-in-docker)
- [Analysing the experiment results](#analysing-the-experiment)
- [Running on your own models and datasets](#running-on-your-own-models-and-datasets)
- [Main changes compared to TalkToModel](#main-changes-compared-to-talktomodel)
- [Citation](#citation)

## Starting the Experiment locally

- create and activate a virtual environment with python 3.9 
  - e.g. ``conda create -n dialogue-xai python=3.9``
- install requirements
- run flask_app.py 
  - While Running the app will display a link to a frontend, this is the old talk-to-model frontend and is currently not
  working since we did not implement the intent recognition yet.
- start [frontend](https://github.com/dimitrymindlin/Measuring-User-Understanding-in-Dialogue-based-XAI-Systems-Frontend?tab=readme-ov-file) and use provided link to start experiment.

## Starting the Experiment in docker

 - make build
 - make run
 - start [frontend](https://github.com/dimitrymindlin/Measuring-User-Understanding-in-Dialogue-based-XAI-Systems-Frontend?tab=readme-ov-file) with docker compose

## Analysing the experiment

- ``experiment_analysis/preprocess_logging_to_analysis_data.py`` is a script to create the analysis_files from logging information in the database.
- We share the preprocessed data in ``experiment_analysis/data_static_interactive``.
- The main analysis is then performed in ``experiment_analysis/analysis_main.py``.

## Running on your own models and datasets

### Model and Dataset
The `data` folder contains the data and train scripts. For example, `adult.csv` is used in `adult_train.py` to train a 
random forest model and save the model, model settings and column information in a separate folder `adult` for the 
explanations later on. When introducing a new dataset, make a new train script and make sure to save the column mappings
and settings in a separate folder.

### Configuration
To run the experiments on your own dataset, create a new config fle in configs folder. Take the `adult-config.gin` as 
an example and adjust the settings to your needs. Then set the `global_config.gin` to your new config file.

## Main changes compared to TalkToModel

- explain/explainers folder. Added explainer classes other than mega_explainer
    - dice_explainer.py
    - anchor_explainer.py
    - ceteris_paribus_explainer.py
    - feature_statistics_explainer.py
- data/response_templates folder to modularize response templates for the different explanations in one place
- additional files to help display correct instances and names for experiment frontend:
    - create_experiment_data to manage experiment flow and instances
    - experiment_helper to manage experiment flow and instances
    - template_manager to handle feature display names
- flask_app.py got many new endpoints to get different datapoints and dataset information as well
  as start new experiments and finish them.
- added adult dataset, preprocessing and model training
- model training now saves mappings for feature preprocessing such as categorical encodings.
- intent recognition (such as T5) is disabled for now and does not work yet in the experiments,
  since we do not need it.
- TalkToModel logging is disabled and experiment logging is done in frontend instead.

## Citation

```bibtex
@inproceedings{DBLP:conf/ecai/MindlinRMC24,
  author       = {Dimitry Mindlin and
                  Amelie Sophie Robrecht and
                  Michael Morasch and
                  Philipp Cimiano},
  title        = {Measuring User Understanding in Dialogue-Based xAI Systems},
  booktitle    = {{ECAI} 2024 - 27th European Conference on Artificial Intelligence,
                  19-24 October 2024, Santiago de Compostela, Spain - Including 13th
                  Conference on Prestigious Applications of Intelligent Systems {(PAIS}
                  2024)},
  pages        = {1148--1155},
  year         = {2024},
  crossref     = {DBLP:conf/ecai/2024},
  url          = {https://doi.org/10.3233/FAIA240608},
  doi          = {10.3233/FAIA240608},
  timestamp    = {Wed, 15 Jan 2025 15:09:06 +0100},
  biburl       = {https://dblp.org/rec/conf/ecai/MindlinRMC24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact

You can reach out to dimitry.mindlin@uni-bielefeld.de with any questions or issues you're running into.
