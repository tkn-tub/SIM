# Smart SIM for DoA estimation
Smart SIM project evaluates an FFT-based direction-of-arrival DoA estimation algorithm with reinforcement learning on the waveform domain.

## Description
SIM (Stacked Intelligent Metasurface) constitutes a MATLAB-based framework that executes the 2D Discrete Fourier Transforms (2D-DFT) explicitly within the waveform domain.
The repository replicates and advances the findings of reference [1] by incorporating a reinforcement learning (RL) component, also via a SIM.
This approach is designed to efficiently localize multiple users within an indoor environment, facilitated by the RL agent.

The system archicture is represented in Fig. 1, where SIM 1 develops the 2D-DFT, and its output is passed to the SIM 2 that estimates the electric angles of arrival.
The architecture operates as follows:

- Mobile users transmit baseband single-carrier pulses modeled as $a(\psi_x,\psi_y)\times s$ over the time interval $T$, where $s$ is constant, and $a(\psi_x,\psi_y)$ represents the Kronecker product of the spatial sequences$e^{j\psi_x(n_x-1)}$ and $e^{j\psi_y(n_y-1)}$, see [1, Eq. (3)-(5)].
The complex exponential indicates the phase introduced by the users' spatial positions, determined by the electric angles $\psi_x$ and $\psi_y$, and indices $n_x$ and $n_y$ referring to the first layer in SIM 1.
- The SIM 1 evaluates the 2D-DFT of the emitted signals by the mobile users.
Its toput evaluates magnitude the of the 2D-DFT and its peaks signals the coordinates of the electrica angles in the $x$ and $y$ axis.
See an example in Fig. 2, as the output produced by SIM 1
- The ouput of SIM 2 provides the estimated angles $\psi_x$ and $\psi_y$ of the peak in the 2D-DFT plain.

<figure>
    <p align="center">
        <img src="https://github.com/tkn-tub/SIM/blob/main/figures/DOA_System_Model.svg?raw=true" alt="nn" width="400">
    </p>
</figure>
<p align="center">
Fig. 1: Representation of the system model with the mobile user (MU) and the two SIMs.
</p>

<figure>
    <p align="center">
        <img src="https://github.com/tkn-tub/SIM/blob/main/figures/DFT_angle_estimation.svg?raw=true" alt="nn" width="400">
    </p>
</figure>
<p align="center">
Fig. 2: 2D-DFT output as derived at the output of the SIM 1.
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