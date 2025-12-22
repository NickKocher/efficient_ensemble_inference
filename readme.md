# Comparative Analysis of Algorithms and Hardware-Level Controls for Resource-Aware AutoML

This is the repository accompanying the code for the master thesis "Comparative Analysis of Algorithms and Hardware-Level Controls for Resource-Aware AutoML" by Janek Paeßens. 

--- 

### Abstract

Automated Machine Learning (AutoML) has become a powerful approach for automating the design and optimisation of machine learning models. As a result, AutoML-generated models are increasingly deployed in real-world applications, often consuming substantial computational resources. Despite this, most current AutoML systems continue to prioritise predictive accuracy as their primary optimisation target and rarely account for the resource footprint of the models they generate. Consequently, AutoML pipelines may produce models that perform well on standard metrics but are impractical for environments with strict running time or energy constraints. In this work, we evaluate and compare various components of AutoML pipelines and analyse how their interactions influence the creation of resource-efficient pools of models. In particular, we examine several single- and multi-objective base-model generation strategies based on hyperparameter optimisation and their interactions with resource-aware and resource-unaware ensembling strategies. We also introduce a novel method, MO-GES, for resource-aware ensembling and compare it against existing approaches demonstrating that MO-GES consistently performs best among all tested ensembling strategies with respect to the hypervolume indicator. Furthermore, we study the effect of hardware-level controls — such as dynamic voltage and frequency scaling and parallelisation — on model resource consumption and develop an allotment technique based on the Non-dominated Sorting Genetic Algorithm II to integrate these controls directly into the AutoML process. Our experiments on the AutoML Benchmark datasets show that the algorithmic and hardware-level controls yield complementary benefits and substantially improve the hypervolume of the pools of solutions discovered by our AutoML framework. Thereby, accuracy-efficiency trade-offs are comprehensively explored. This enables practitioners to systematically select models that align with their specific computational and resource constraints

--- 

### Repository Structure

The repository is structured as follows. 

- **./analysis/** contains code to 
    - extract results from the conducted experiments into a few files (./analysis/collect_scripts/)
    - analyse the extracted results and re-create the figures used in the thesis (./analysis/notebooks/)
- **./auto_sklearn_search_space/** and **./auto_sklearn_search_space_no_freq/** contain code for the AutoSklearn Search Space (with and without frequency scaling hyperparameters respectively). This code was not used for final experiments. Thus, full out-of-the-box compatability with the AutoML framework may not be guaranteed.
- **./experiments/** contains the code to run the experiments conducted within the master thesis. In particular,
    - analysis_gbmlp.py was used to generate the data for the frequency scaling analysis on the Intel Xeon Platinum 8480+
    - collect_ens_preds.py was used to extract ensembling predictions of base models and ensembles in order to restrict the evaluation on the test data to the corresponding pareto fronts of the solution pools using the ensembling data, thereby accounting for overfitting
    - run_bmg.py runs the base model generation including refits for ensembling evaluations and final evaluations.
    - run_create_preds.py runs the creation of predictions for ensembling predictions on all hardware-level setups (i.e. frequency scaling vs no frequency scaling and at different core counts)
    - run_ensembling.py runs all ensembling and allotment experiments based on prior base model generation
- **./frequency/** contains code to set the frequency of the CPU correspondingly
- **./mosmac3/** is adapted from the code for MO-SMAC based on the framework proposed by [Rook et al.](https://pubmed.ncbi.nlm.nih.gov/40096519/) and lays foundation for the multi-objective Bayesian optimisation used within the thesis.
- **./phem/** contains code adapted from [Purucker et al.](https://github.com/Atraxus/phem/tree/804fc55c5f98f1d52e7d437c37f989ff734cfbbc) that comprises important utility features and algorithmic implementations for ensembling.
- **./raaml/** contains the actual framework. This directory is further divided into 
    - ./raaml/base_model_generation/ comprises implementations and utility for the base model generation methods used within this thesis
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