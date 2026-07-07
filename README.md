# UAVUncertain
The data and code for the article SDR-LSAC

This repository contains the official implementation of SDR-LSAC, a robust and adaptive deep reinforcement learning framework designed for unmanned aerial vehicle (UAV) trajectory tracking under severe wind gusts, variable payload, and execution/communication latencies

Key Features

1. Dual-Channel Decoupled Context Encoder: Explicitly isolates historical state and action trajectories to resolve systemic latency and partial observability in POMDPs.
2. Asymmetric Prioritized Experience Replay: Implements an asymmetric TD-error penalty mechanism to mitigate Q-value overestimation and catastrophic exploration crashes.
3. Curriculum-Driven Soft Decay: Automatically scales down historical priorities during curriculum transition to stabilize learning across escalating environmental difficulties.
4. High-Fidelity Evaluation: Integrates real-world OpenStreetMap road network constraints with dynamic aerodynamic models (blade flapping, wind-sail torques, and quad drag asymmetry).

System Configuration & Hardware
The codebase has been successfully developed, trained, and tested under the following environment:
Operating System: Windows 10/11 / Ubuntu 20.04 LTS
CPU: AMD Ryzen 7 5800H (8 Cores, 16 Threads)
GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
Python Version: Python 3.7 (Stable environment)

Installation & Dependencies
To set up the workspace and install the required Python libraries, please execute the following command:
pip install torch numpy scipy matplotlib seaborn pandas osmnx networkx gym

Core Dependency Breakdown:
torch: Deep learning backend used to implement the actor, critic, and context encoder networks.
numpy: Fast numerical calculations, matrix operations, and coordinate transformations.
scipy: Used for cubic B-spline path smoothing, Savitzky-Golay telemetry filtering, 2D wind field interpolation, and statistical hypothesis testing (Welch's t-test).
matplotlib & seaborn: Plotting the paper's core figures, including 2D spatial trajectories, attitude/delay dual-Y telemetry, split-violin plots, and 2D safety-boundary heatmaps.
pandas: Structured data processing for Seaborn multi-distribution metrics.
gym: Core base class for the custom-built high-fidelity drone dynamics simulator.
osmnx & networkx: Downloading, UTM-projecting, and shortest-path routing of the real-world Shanghai logistics map.

Project Structure
├── logs/                      # Training logs and raw tensor evaluation curves
├── models/                    # Pre-trained neural network weights (.h5 files)
├── .gitattributes             # Git attributes config
├── .gitignore                 # Git ignore config
├── BasicEnv70.py              # Custom UAV dynamics environment & POMDP wrapper
├── Map0.py                    # OpenStreetMap road network download & path routing
├── MetaSAC.py                 # Core AER-MDSAC algorithm implementation
├── README.md                  # Project documentation (this file)
├── run_all_experiments.py     # Main script to run Monte-Carlo tests and OOD sweeps
└── Show_results.py            # Academic visualization script to generate figures and tables

How to Run
1. Robustness Testing and Data Collection
To run the evaluation benchmarks (including Monte-Carlo runs for Table 3/6 and Out-of-Distribution boundary sweeps), execute:
python run_all_experiments.py
This script will load the pre-trained weights from models/ and output raw data .npy files into the results/ folder (the folder is automatically created if it does not exist).

2. Paper Figure and Table Generation
Once the raw .npy files are populated, execute the visualization script to generate the exact figures and statistical tables presented in the manuscript:
python Show_results.py

This script will produce:
Figure 6: Convergence curves with 95% Confidence Intervals under curriculum transitions.
Figure 7: 2D top-down trajectories overlaid on the OSM map with local inset magnifiers.
Figure 8: Chronological telemetry and 2D interpolated wind field response plots.
Table 1 & 2: Grayscale-aligned LaTeX tables presenting mean, standard deviation, and Welch's t-test p-values.

License
This project is licensed under the MIT License - see the LICENSE file for details.