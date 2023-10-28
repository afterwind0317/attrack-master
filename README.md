# Crowdsourced Attack Experiment Source Code

# Table of Contents

- [Project Structure](#Project Structure)
- [Experiment Introduction](#Experiment Introduction)

```sh
git clone https://git....
```

## Project Structure
```
PLT
├── experience_plt_0
│  ├── s4-dog
│  └── synthetic
├── experience_plt_1
│  ├── s4-dog
│  └── synthetic
├── experience_plt_2
│  ├── s4-dog
│  └── synthetic
├── experience_plt_3
│  ├── s4-dog
│  └── synthetic
├── experience_plt_4
│  ├── s4-dog
│  └── synthetic
├── experience_plt_5
│  ├── s4-dog
│  └── synthetic
└── experience_plt_6
   ├── s4-dog
   └── synthetic

s4_Dog
├── baseline
│  ├── CP
│  └── FC
├── data
├── experience
│  ├── acc_result
│  ├── acc_result_del
│  ├── result
│  └── utils
│     ├── ACC
│     ├── Attack
│     ├── Delete
│     ├── KCDN_W
│     ├── percentage
│     └── Top
└──preData

synthetic
├── baseline
│  ├── CP
│  └── FC
├── data
├── experience
│  ├── acc_result
│  ├── acc_result_del
│  ├── result
│  └── utils
│     ├── ACC
│     ├── Attack
│     ├── Delete
│     ├── KCDN_W
│     ├── percentage
│     └── Top
└──preData
```

## Experiment Introduction
The experiment is divided into three files: PLT, s4-dog, and synthetic. 
PLT is used for plotting statistical graphs of the experiment results, 
while the latter two files are experiments involving two crowdsourced datasets, 
including the real dataset s4-dog and the synthetic dataset synthetic. 
The PLT file contains the statistical results of six experiments, which we will introduce one by one.


### PLT
This file contains statistical graphs for 6 experiments, with "experiment_0" focusing on the impact of worker capabilities on repeated answers.

"experiment_0": The file contains statistical graphs for six experiments, with Experiment 0 focusing on the impact of worker capabilities on repeated answers.

"experiment_1": Changes in worker earnings and ability distribution under collusion.

"experiment_2": Hyperparameter experiments.

"experiment_3": A chart showing the impact of collusion on the accuracy of the original dataset.

"experiment_4": A demonstration of the change in accuracy rates for collusion detection experiments.

"experiment_5": Top-k experiment.

"experiment_6": Presentation of the filtering experiment.

If you need further assistance or have more content to translate, please let me know.


### s4-dog
This file includes four subdirectories: "baseline", "data", "experience" and "preData".

The 'baseline' directory represents comparative experiments corresponding to two collusion detection methods, CP and FC.

The 'data' directory contains datasets in the form of triplets (question, worker, answer) with 'answer_file.csv' and correct answers in 'result_file.csv'.

The 'experience' directory serves as the main storage for experimental results, including 'acc_result' and 'acc_result_del', as well as scripts for collusion experiments in 'result' and utilities for detecting collusion in all experiments.

These two directories respectively summarize collusion experiments and filtering experiments.

The 'result' directory includes experimental data for collusion detection at ratios of 0.6, 0.7, 0.8, 0.9, and 1.0.

In the 'result' directory with a collusion detection ratio of 0.8, it contains experiments with different collusion worker ratios, including 10%, 20%, 30%, 40%, and 50% of all workers, and also includes top-k experiments.

Finally, the 'preData' folder contains script files for data preprocessing experiments. In these scripts, the dataset's order is shuffled to generate crowdsourced datasets with completely different answer orderings for collusion and detection experiments

### synthetic
The experiment structure is consistent with the s4-dog file.