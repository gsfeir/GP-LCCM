<h1 style="font-family: 'Arial', sans-serif;">Gaussian Process – Latent Class Choice Model (GP-LCCM)</h1>

This repository contains the code for estimating Gaussian Process – Latent Class Choice Models (GP-LCCM) using the Expectation Maximization (EM) algorithm. The model employs a Gaussian Process Classifier to flexibly model class membership probabilities and uses class-specific weighted multinomial logit models to explain choice behavior.


## Overview
The GBM_LCCM code: 
- **Processes Data:** Constructs sparse matrices from long-format data that map alternatives, observations, and decision-makers. It also allows for imposing analyst-defined choice set constraints.
- **Class Membership Modeling via Gaussian Process:** Uses a Gaussian Process Classifier (from scikit-learn) to estimate the probability that each individual belongs to a latent class. The classifier returns both probability estimates and a log marginal likelihood.
- **Class-Specific Choice Modeling:** Implements a weighted multinomial logit model for each latent class to model the choice behavior of individuals. It uses numerical optimization (BFGS) to update the parameters and compute standard errors.
- **Expectation Maximization (EM) Algorithm:** Alternates between calculating latent class responsibilities (E-step) and updating model parameters (M-step) until convergence.
- **Outputs Results:** Displays model fit statistics (log-likelihood, AIC, BIC), parameter estimates (with standard errors, t-statistics, and p-values), and optionally exports prediction enumerations.

## Requirements
- Python 3.7+ (tested with Python 3.12)

## Usage 
For an example of how to use this code, refer to the Jupyter Notebook: GP_LCCM_Example.ipynb.

## Configuration
- **Gaussian Process Kernel:** The code uses a default kernel (a dot product) from scikit-learn. You can modify the kernel settings near the top of the code as needed.
- **EM Algorithm Settings:** Adjust tol (tolerance) and max_iter (maximum iterations) to control convergence.
- **Initialization:** Choose between 'random' or 'kmeans' initialization for the GMM parameters (GMM_Initialization).

## Contributing
Contributions, bug fixes, and feature suggestions are welcome. Please open an issue, submit a pull request, or feel free to contact me directly.

## Acknowledgements
- **Author:** Georges Sfeir.
- **Advising:** Filipe Rodrigues.
- **Based On:** The latent class choice model (lccm) package and some function from the GaussianProcessClassifier class of sklearn.

## More Information
For more information about the GP_LCCM model see the following paper:
 
Sfeir, G., Rodrigues, F., Abou-Zeid, M. (2022). “Gaussian Process Latent Class Choice Models.” Transportation Research Part C: Emerging Technologies, 136, 103552. https://doi.org/10.1016/j.trc.2022.103552

## Citation
If you find this model useful in your research or work, please cite it by citing the paper above.
