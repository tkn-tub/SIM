# Reinforcement learning component
This code implements a DQN-based RL agent whose Q-network is physically embodied in SIM 2

## Description
This code implements a DQN-based where the RL agent whose Q-network is physically embodied in SIM 2, and the optimal policy is implemented digitally with the pre-computed values by the SIM 2, see Fig. 1 below.
The metasurface layers themselves act as the network, so the forward pass is performed by optical propagation through SIM 2 rather than by digital compute.
Within this architecture, the definition of states and actions follows the description

- **State** $s_t$: Given by the 2D-DFT plane $\mathfrak{F}(t)$ produced by SIM 1 and observed
  at the input layer of SIM 2. Each state is a snapshot of the angular spectrum
  in the $(\psi_x, \psi_y)$ grid, where peaks indicate the directions of arrival
  of the mobile users at time $t$.
- **Action** $a_t = (\Delta\psi_x, \Delta\psi_y)$: Is formulated as the discrete increment applied
  to the phase-shift configuration $\xi_{0,t}$ of the first layer of SIM 1.
  The action set is the finite grid of admissible $(\Delta\psi_x, \Delta\psi_y)$
  pairs that steer the observation window to track the users between snapshots.
- **Reward** $r_t = r(\Delta\psi_x, \Delta\psi_y)$: *[describe your reward here —
  e.g., the received energy at the predicted user bin, or the negative tracking
  error between the predicted and observed peak positions at $t+1$].*
- **Policy** $\pi(a_t \mid s_t)$: $\varepsilon$-greedy over the action-value
  function $Q(s_t, a_t)$.
  The $Q(s_t, a_t)$ is approximated by the neural network embedded in SIM 2.
  With probability $(1-\varepsilon)$ the agent selects
  $a_t = \arg\max_{a} Q(s_t, a)$; with probability $\varepsilon$ it samples
  uniformly from the action set to encourage exploration.
  The value of $\varepsilon$ decays over training episodes.

The RL agent is implemented within the file ![MATLAB](docs/matlab_icon.png) [`SIM_1_SIM_2_DQN_AI.mlx`](SIM_1_SIM_2_DQN_AI.mlx).
<!--[<img src="docs/matlab_icon.png" alt="MATLAB" width="16"/> `SIM_1_SIM_2_DQN_AI.mlx`](SIM_1_SIM_2_DQN_AI.mlx).-->

The system archicture is represented in Fig. 1, where SIM 1 develops the 2D-DFT, and its output is passed to the SIM 2 that estimates the electric angles of arrival.
The architecture operates as follows:

- Mobile users transmit baseband single-carrier pulses modeled as $\boldsymbol{a}(\psi_x,\psi_y)\times s$ over the symbol time-interval $T_s$, where $s$ is constant, and $\boldsymbol{u}(\psi_x,\psi_y)$ is a vector that represents the Kronecker product of the spatial sequences$e^{j\psi_x(n_x-1)}$ and $e^{j\psi_y(n_y-1)}$, see [1, Eq. (3)-(6)].
The complex exponential indicates the phase introduced by the users' spatial positions, determined by the electric angles $\psi_x$ and $\psi_y$, and indices $n_x$ and $n_y$ referring to the first layer in SIM 1.
- The received signal at the first layer of the SIM 1 is modeled with a clustered-delay-line (CDL) channel.
Specifically, to model indoor scenarios in industrial environments.
- The SIM 1 evaluates the 2D-DFT of the emitted signals by the mobile users.
Its toput evaluates magnitude the of the 2D-DFT and its peaks signals the coordinates of the electrica angles in the $x$ and $y$ axis.
See an example in Fig. 2, as the output produced by SIM 1.
The SIM 1 operates as indicated in [1, Sec. III], where three main parameters are defined $N$, which is the number of elements in the first layer, and $T$, which is the total number of time slots where the 2D-DFT is evaluated.
- The SIM 2 is interconnected to the output of the SIM 1 and develops a fullly-connected layer neural network (NN).
The ouput of SIM 2 provides the estimated angles $\psi_x$ and $\psi_y$ of the peak in the 2D-DFT plain.
That is the values for $\hat\psi_x$ and $\hat\psi_y$.

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
This code is tested in MATLAB 2025a, and the required toolboxes are listed in the table below.

| Matlab Toolbox  | Version |
| ------------- | ------------- |
| Signal Processing Toolbox | 25.1  |
| Communications Toolbox | 25.1  |
| Phased Array System Toolbox  |25.2|
| WLAN Toolbox | 25.1  |
| 5G Toolbox  |25.2|

## Usage

This project directly runs from the Matlab accesible on each folder within the folder code [code/](https://github.com/tkn-tub/SIM/tree/main/code).
Within this folder you find the following ones:

📁 [DoA/](https://github.com/tkn-tub/SIM/tree/main/DoA): Includes the code to estimate the electric angles $\psi_x$ and $\psi_y$ of a single user using the SIM 1.
In this code the received signal model follows the linear transformation model $\sqrt{\rho}\boldsymbol{a}(\psi_x,\psi_y)\times s+\boldsymbol{u}$, where $\rho$ refers to the signal-to-noise ratio (SNR) and $u$ refers to a CSCG random vector of average value zero and variance $1$. 

📁 [Channel Model/](https://github.com/tkn-tub/SIM/tree/main/Channel_Model):  This folder includes a clustered-delay-line (CDL) model for the received signal, which mimics indoor industrial scenarios.


## Features
- **Modeling the Wave-Domain Computing:**  The stacked metasurface performs computation as EM waves propagate, without digital hardware.
- **DoA Estimation:**  This code develops a direct mapping between direction-of-arrival and SIM output intensities.

## Contributing
Interested contributors can contact the project owners.
Please refer to the Contact Information below. We identify further developments for more complex scenarios like estimating the distance to multiple cancer cells.

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