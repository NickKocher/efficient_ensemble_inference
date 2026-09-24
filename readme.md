# Efficient Ensemble Inference
## Paper

For your convenience, the paper is available in Github:

https://github.com/NickKocher/efficient_ensemble_inference/blob/automl26/Efficient_ensemble_inference-cr.pdf

---
This is the repository accompanying the code for the AutoML submission "Efficient Ensemble Inference". 

--- 

### Abstract

Ensembling improves predictive performance in AutoML systems but often leads to inefficient inference on CPUs,
as most frameworks execute ensemble members sequentially while 
allocating all available CPU cores to each model. Since inference scales poorly beyond a 
few cores and ensemble members vary in running time, this wastes both time and energy. 
We address this by framing ensemble inference as a moldable task scheduling problem, 
where CPU core allocation and execution order for inference time and energy consumption 
are jointly optimised. Building on a two-stage allotment and scheduling framework, we 
introduce NSGA-P, a multi-objective allocation strategy based on NSGA-II as an analytical 
tool to understand the benefits of efficient ensemble inference using offline resource measure- 
ments. We evaluate our approach on 104 datasets from the AMLB benchmark suite. NSGA-P 
outperforms the naïve maximum-cores baseline used by existing AutoML frameworks in 
over 96% of ensembles. NSGA-P allotments halve inference time on 8 cores and achieve 
up to a 74% reduction in both inference time and energy consumption on 64 cores. These 
results demonstrate that intelligent parallel resource allocation can substantially improve 
the efficiency of resource-aware AutoML systems.

--- 
### Installation
Installation requires Python 3.11 with GCCcore 13.2.0 and SWIG 4.1.1.

Install requirements from requirements.txt

Install package using `pip install -e .`


### Example Usage
We provide anonymised Slurm scripts in ./slurm_scripts to rerun all experiments in the paper.

Scripts are numbered according to the execution order of required.

For a minimal usage example, restrict the slurm array to the first job.

This will execute one job for one dataset on one seed.

### Hardware dependent changes
The experiments in this repository measure energy on hardware devices using the amd_energy driver with elevated access.

To allow for execution on other hardware we disabled the interface in the ./raaml/resource_provider.py file.

To use the amd_energy driver please install from https://github.com/amd/amd_energy.

and revert the changes made to the AMDEnergyProvider class in ./raaml/resource_provider.py

### Repository Structure

The repository is structured as follows. 

- **./analysis/** contains code to 
    - extract results from the reproduced experiments into a few files (./analysis/collect_scripts/)
    - analyse the extracted results  
    - and use available data in ./analysis/data/ to re-create the figures used in the paper (./analysis/notebooks/).
- **./experiments/** contains the code to run the experiments. In particular,
    - analysis_gbmlp.py was used to generate the data for the consistency analysis visualised in Figure 2 and the tradeoff analysis visualised in Figure 1.
    - run_bmg.py runs the base model generation including refits for ensembling evaluations and final evaluations.
    - run_create_preds.py runs the creation of predictions for ensembling predictions on all hardware-level setups
    - collect_ens_preds.py was used to extract ensembling predictions of base models and ensembles in order to restrict the evaluation on the test data to the corresponding pareto fronts of the solution pools using the ensembling data, thereby accounting for overfitting
    - run_ensembling.py runs all ensembling and allotment experiments based on prior base model generation
- **./raaml/** contains the actual framework. This directory is further divided into 
    - ./raaml/base_model_generation/ comprises implementations and utility for the base model generation method
    - ./raaml/config_spaces/ contains the code to create the configuration space based on XGBoost, RealMLP, TabM
    - ./raaml/ensembling/ comprises code related to ensemble selection, scheduling and allotment and base model pruning
    - ./raaml/util/ provides various utility implementations used within the framework
    - ./raaml/raaml.py is the actual implementation of the framework and assembles all base model generation, ensembling, allotment and scheduling features
    - ./raaml/resource_provider.py provides the baseline for measuring resource consumptions on the used platforms

---

### Used Software and Dependencies

- Python 3.11.5 with GCCcore 13.2.0 was used for all experiments
- The used python packages and their versions are detailed in requirements.txt
- Additionally, SMAC requires SWIG as a dependency. We have used SWIG 4.1.1 for all experiments
