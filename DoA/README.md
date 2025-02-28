# Distance Estimator
This project develops a neural network (NN) module for detecting the distance among cells.

## Description
This project develops a NN module to estimate the distance among cells. The project features the application where the distance to a tumor cell is estimated based on the vesicles-amount observed at the immune cell; see Fig. 1 below. Cancer and immune cells exchange vesicles in close proximity to each other, and the concentration level of vesicles can be used to estimate distance as it is a distance-dependent process.

<figure>
    <p align="center">
        <img src="https://github.com/tkn-tub/NN_molecular_communications/blob/main/Figures/distance_estimator.png?raw=true" alt="nn" width="500">
    </p>
</figure>
<p align="center">
Fig. 1: Components for estimating the distance among cell using a feedforward NN.
</p>

The dataset is created with the code provided by the authors in [1] comprising the number of released vesicles at the Immune cell with time, as illustrated in Fig. 1a. This number of vesicles depends on the distance to the tumor cell, which is the parameter to estimate The dataset is processed to evaluate the slope, see the subfigure in Fig. 1a, from the slope the peak amplitude and location are taken as the two features to train and deploy the neural network.

The NN is a low-complexity feedforward architecture implemented in MATLAB and comprises a single layer and two nodes. The output of the NN is the predicted distance, as illustrated in Fig. 1c, the model devise a quite accurate estimator.

## Installation
This code is tested in MATLAB 2023b, and the required toolboxes are listed in the table below.

| Matlab Toolbox  | Version |
| ------------- | ------------- |
| System Identification Toolbox  | 23.2  |
| Deep Learning Toolbox  | 23.2  |
|Statistics and Machine Learning Toolbox|23.2|

## Usage

This project directly runs from the file `A_Master_File.mlx`, where the NN model is trained and deployed. This file calls to the other two project files `Parameters.mlx` and optionally to `Dataset_compiler.mlx`. By default, the code loads the stored file `Dataset_cell2cell.mat`, accesible on the [IEEE DataPort portal at this link](https://ieee-dataport.org/documents/dataset-cell-cell-communications) after loggin.

## Features
- **Realistic model for vesicles exchange among cells:** This code evaluates a realistic model for the exchange of molecules between immune and cancer cells. The code within the file `Dataset_compiler.mlx`, originally provided by Mohammad Zoofaghari, follows the mathematical developments in [1].
- **Low-complex distance estimator to a cancer cell:** This solution features a 2 nodes NN to accurately estimate the distance to a cancer cell from a neighbord immune cell.

## Contributing
Interested contributors can contact the project owner. Please refer to the Contact Information below. We identify further developments for more complex scenarios like estimating the distance to multiple cancer cells.

## License
![Licence](https://img.shields.io/github/license/larymak/Python-project-Scripts)

## Acknowledgements
We want to acknoledge the support provided by Mohammad Zoofaghari, author of the paper in [1] for giving us the code to generate the dataset.

## References
<a name="fn1">[1]</a>: M. Zoofaghari, F. Pappalardo, M. Damrath, and I. Balasingham, “Modeling Extracellular Vesicles-Mediated Interactions of Cells in the Tumor Microenvironment,” IEEE Transactions on NanoBioscience,
vol. 23, no. 1, pp. 71–80, Jan. 2024. [Link](https://ieeexplore.ieee.org/document/10149035)

## Contact Information

- **Name:** Jorge Torres Gómez

    [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github)](https://github.com/jorge-torresgomez)

    [![Email](https://img.shields.io/badge/Email-jorge.torresgomez@ieee.org-D14836?logo=gmail&logoColor=white)](mailto:jorge.torresgomez@ieee.org)

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-torresgomez-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/torresgomez/)

    [![Website Badge](https://img.shields.io/badge/Website-Homepage-blue?logo=web)](https://www.tkn.tu-berlin.de/team/torres-gomez/)