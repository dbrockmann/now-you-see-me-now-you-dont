## Hyperparameters of Classifiers, Optimized using Randomized Search

| Classifier | Parameter                  | UNSW-NB15       | CIC-IDS2017     | CSE-CIC-IDS2018 | WEB-IDS23       |
|------------|----------------------------|-----------------|-----------------|-----------------|-----------------|
| **MLP**    | Normalization              | min-max         | quantile        | min-max         | quantile        |
|            | Hidden layer sizes         | 256, 128, 64    | 256, 128, 64    | 256, 128        | 256, 128, 64    |
|            | Learning rate              | 0.001749        | 0.0001          | 0.000373        | 0.000254        |
| **CNN**    | Normalization              | quantile        | min-max         | min-max         | min-max         |
|            | Dropout                    | 0.279164        | 0.158049        | 0.002158        | 0.293439        |
|            | Filter sizes               | 32, 64          | 32, 64, 64      | 32, 64          | 32, 64, 64      |
|            | Hidden layer sizes         | 256, 128        | 256             | 256, 128        | 256             |
|            | Kernel size                | 5               | 3               | 3               | 5               |
|            | Learning rate              | 0.000148        | 0.000107        | 0.000131        | 0.000162        |
| **AE**     | Normalization              | quantile        | quantile        | quantile        | quantile        |
|            | Encoder hidden layer sizes | 256             | 256, 128, 64    | 256, 128, 64    | 256             |
|            | L1 activation penalty      | 0.001106        | 0.00012         | 0.000257        | 0.000208        |
|            | Latent size ratio to input | 75.67%          | 31.27%          | 59.37%          | 75.55%          |
|            | AE learning rate           | 0.000509        | 0.002469        | 0.000476        | 0.000612        |
|            | MLP hidden layer sizes     | 256, 128, 64    | 256             | 256, 128, 64    | 256             |
|            | MLP learning rate          | 0.000588        | 0.000371        | 0.000172        | 0.000817        |
| **RF**     | Normalization              | quantile        | quantile        | min-max         | quantile        |
|            | Criterion                  | entropy         | gini            | entropy         | entropy         |
|            | Maximum depth              | 29              | 26              | 28              | 20              |
|            | Number of trees            | 146             | 104             | 106             | 135             |
| **SVM**    | Normalization              | quantile        | quantile        | quantile        | quantile        |
|            | Kernel gamma               | 1               | 0.1             | 0.1             | 0.1             |
|            | Kernel C                   | 10              | 10              | 0.1             | 0.1             |
|            | Alpha learning rate        | 0.000032        | 0.000001        | 0.000002        | 0.000111        |
|            | Loss function              | modified huber  | modified huber  | modified huber  | hinge           |
| **KNN**    | Normalization              | min-max         | quantile        | min-max         | min-max         |
|            | Number of neighbors        | 5               | 3               | 5               | 5               |
