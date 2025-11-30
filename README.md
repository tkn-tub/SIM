# Smart SIM for DoA estimation
Smart SIM project evaluates an FFT-based DoA estimation algorithm with reinforcement learning on the waveform domain.

## Description
SIM (Stacked Intelligent Metasurface) is a MATLAB-based framework that implements the 2D Discrete Fourier Transforms (2D-DFT) directly in the waveform domain.
The repository reproduces and extends results from ref. [1] by including a reinforcement learning (RL) module also through a SIM.
This solutions aims to fast localize multiple users in an indoor scenario.

The system archicture is represented in Fig. 1, where mu

<figure>
    <p align="center">
        <img src="https://github.com/tkn-tub/SIM/blob/main/figures/DOA_System_Model.SVG?raw=true" alt="nn" width="200">
    </p>
</figure>
<p align="center">
Fig. 1: Representation of the system model with the mobile user (MU) and the SIM.
</p>


## Installation
This code is tested in MATLAB 2023b, and the required toolboxes are listed in the table below.

| Matlab Toolbox  | Version |
| ------------- | ------------- |
|  | 23.2  |
|  | 23.2  |
|  |23.2|

## Usage

This project directly runs from the file `A_Master_File.mlx`, where .

## Features
- **Evaluation of the SIM performance:** This code evaluates a .


## Contributing
Interested contributors can contact the project owner. Please refer to the Contact Information below. We identify further developments for more complex scenarios like estimating the distance to multiple cancer cells.

## License
![Licence](https://img.shields.io/github/license/larymak/Python-project-Scripts)

## Acknowledgements
This project was supported in part by the Federal Ministry of Education and Research (BMBF, Germany) within the 6G Research and Innovation Cluster 6G-RIC under Grant 16KISK020K..

## References
<a name="fn1">[1]</a>: J. An et al., "Two-Dimensional Direction-of-Arrival Estimation Using Stacked Intelligent Metasurfaces," in IEEE Journal on Selected Areas in Communications, vol. 42, no. 10, pp. 2786-2802, Oct. 2024. [Link](https://ieeexplore.ieee.org/document/10557708)

## Contact Information

- **Name:** Jorge Torres Gómez

    [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github)](https://github.com/jorge-torresgomez)

    [![Email](https://img.shields.io/badge/Email-jorge.torresgomez@ieee.org-D14836?logo=gmail&logoColor=white)](mailto:jorge.torresgomez@ieee.org)

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-torresgomez-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/torresgomez/)

    [![Website Badge](https://img.shields.io/badge/Website-Homepage-blue?logo=web)](https://www.tkn.tu-berlin.de/team/torres-gomez/)

- **Name:** Karel Toledo de la Garza

    [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github)](https://github.com/kareltdlg)

    [![Email](https://img.shields.io/badge/Email-karel.toledo@usach.cl-D14836?logo=gmail&logoColor=white)](mailto:karel.toledo@usach.cl)

    [![LinkedIn](https://img.shields.io/badge/LinkedIn-kareltoledo-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/karel-toledo-de-la-garza-a38ab6a1/)

    [![Website Badge](https://img.shields.io/badge/Website-Homepage-blue?logo=web)](https://die.usach.cl/karel-toledo-delagarza/)