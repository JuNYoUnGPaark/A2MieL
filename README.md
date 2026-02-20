# Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition


<p align="center"><img src='./overall.png'></p>

This repository implements the methodology proposed in the paper "Physics-Derived Rate and Phase Decomposition with Attention-to-Attention Reconciliation for Robust Inertial Activity Recognition".


## Paper Overview
**Abstract**: 

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
