
# now-you-see-me-now-you-dont

This repository provides the code base for our scientific work on constrained adversarial attacks in NIDS with a comparison of datasets and classification methods.

It is structured in [flows](flows/), covering an operation that produces or/and consumes artifacts (e.g., preprocessing data or training the models), which can be called using `python main.py <flow-name>` (see [main.py](main.py)). All code regarding data processing can be found in [datasets](datasets/) and [preprocessing](preprocessing/), training and testing classifiers in [classification](classification/) and [models](models/) and applying and evaluating adversarial attacks in [attacking](attacking/) and [attacks](attacks/). Necessary parameters, including the models and datasets used in the run of a flow, are defined in [parameters.py](parameters.py).

## Usage

```
pip install -r requirements.txt
python main.py <flow-name>
```

`<flow-name>` describes the processing step:
- `data-cleaning`: Pre-processing of the datasets
- `data-undersampling`: Tests undersampling strategies
- `data-feature-selection`: Tests feature selection parameters
- `data-oversampling`: Tests oversampling strategies
- `model-parameter-fitting`: Finds the optimal model hyper-parameters using cross validation
- `model-retraining`: Retrains the models with hyper-parameters on the full training set
- `model-evaluation`: Evaluates the models on the test set
- `attack-initialization`: Initializes the adversarial attack samples
- `attack-minimization'`: Minimizes successful adversarial examples
- `attack-evaluation-initialization`: Evaluates the attack success
- `attack-evaluation-minimization`: Evaluates the perturbation minimization

Define the datasets, models and attack types of a run in [parameters.py](parameters.py). The paths to the raw datasets must be defined in the respective loading scripts in [datasets](datasets/). To add new datasets, models or attacks, define them in the respective folder and load them in [parameters.py](parameters.py).
