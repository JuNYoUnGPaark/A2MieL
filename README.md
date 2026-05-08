# Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition


<p align="center"><img src='./overall.png'></p>

This repository implements the methodology proposed in the paper "Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition".


## Paper Overview
**Abstract**: Human activity recognition (HAR) with wearable
inertial sensors requires models that are not only accurate
but also less sensitive to common signal-level perturbations and efficient for practical inference. However, many
existing deep-learning HAR methods fuse motion cues in a
static manner, although first-order dynamics and secondorder structural changes provide complementary information and may respond differently under signal-level perturbations. To address this issue, we propose an energyinspired landscape modeling framework with an Attentionto-Attention (A
2
) reconciliation mechanism. The proposed
framework introduces energy-based inductive biases motivated by kinetic and potential energy concepts to regularize the latent representation of inertial signals and to
derive complementary rate and phase feature views. The A
2 module then separates energy-variation cues into raterelated and curvature-related proxies, and dynamically reconciles the two views by estimating cue reliability from the
joint interaction between attention responses and feature magnitudes. Experiments on five public HAR benchmarks,
including UCI-HAR, WISDM, MotionSense, MHEALTH, and PAMAP2, show that the proposed method provides competitive
recognition performance while maintaining a compact model size and low inference cost. In addition, it shows reduced
sensitivity under the evaluated controlled perturbation settings and practical inference efficiency, with approximately
0.07M parameters, an inference latency below 3.92 ms on a desktop CPU, and 18.88 ms on a Raspberry Pi 4B.

## Dataset
- **UCI-HAR** dataset is available at _https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones_
- **PAMAP2** dataset is available at _https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring_
- **MHEALTH** dataset is available at _https://archive.ics.uci.edu/dataset/319/mhealth+dataset_
- **WISDM** dataset is available at _https://www.cis.fordham.edu/wisdm/dataset.php_
- **MotionSense** dataset is available at _https://github.com/mmalekzadeh/motion-sense?tab=readme-ov-file_

## Requirements
```
torch==2.5.0+cu126
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
matplotlib==3.10.0
seaborn==0.13.2
fvcore==0.1.5.post20221221
```
To install all required packages:
```
pip install -r requirements.txt
```

## Codebase Overview
- `model.py` - Implementation of the proposed **Energy-Inspired Landscape Modeling** framework.
The implementation uses PyTorch, Numpy, pandas, scikit-learn, matplotlib, seaborn, and fvcore (for FLOPs analysis).

## Citing this Repository

If you use this code in your research, please cite:

```
@article{Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition},
  title = {Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition},
  author={JunYoung Park and Myung-Kyu Yi}
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}
```

## Contact

For questions or issues, please contact:
- JunYoung Park : park91802@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
