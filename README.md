# Pan-cancer Outcome Prediction using Multi-classifier Systems

Code repository for the ACM-BCB 2025 short paper ([https://doi.org/10.1145/3765612.3767240](https://doi.org/10.1145/3765612.3767240)).

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/sakhawatsaimon/Pancancer-MCS/raw/main/misc/overview.svg)

## Requirements

Python: `requirements.txt`  
MATLAB:
- MATLAB version: 24.1.0.2578822 (R2024a) Update 2
- Additional packages: [MALSAR 1.1](http://www.yelabs.net/software/MALSAR/)

## Data

Please contact authors for the datasets.

## Experiments

The following 3 scripts are used to conduct experiments:

- `experiment.py` (python): evaluate performance of MCS and baseline models. 
- `experiment_cancer_aware.py` (python): evaulate performance of additional cancer-aware baselines.
- `experiment_mtl.m` (MATLAB): evaluate multi-task learning moodel performance.

Model predictions are stored in `results\pred_probability`. To get predictions for individual samples using MCS and baseline models, please re-run the script `experiment.py`.

## Evaluation

To generate results presented in the paper, run this command:

```
python plots.py
```


## License

This work is licensed under the [MIT License](https://github.com/sakhawatsaimon/Pancancer-MCS/blob/main/LICENSE).