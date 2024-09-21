
# now-you-see-me-now-you-dont

This repository provides the code base for our scientific work on constrained adversarial attacks in NIDS with a comparison of datasets and classification methods.

It is structured in [flows](flows), covering an operation that produces or/and consumes artifacts (e.g., preprocessing data or training the models), which can be called using `python main.py <flow-name>` (see [main.py](main.py)). All code regarding data processing can be found in [datasets](datasets) and [preprocessing](preprocessing), training and testing classifiers in [classification](classification) and [models](models) and applying and evaluating adversarial attacks in [attacking](attacking) and [attacks](attacks). Necessary parameters, including the models and datasets used in the run of a flow, are defined in [parameters.py](parameters.py).
