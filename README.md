# DDBA
[Guorui Li, Runxing Chang, Ying Wang, Cong Wang. Dual-domain based backdoor attack against federated learning. Neurocomputing, 2025, 129424: 1-13.](https://www.sciencedirect.com/science/article/pii/S0925231225000967)

## Installation
Install Pytorch

## Usage
### Prepare the dataset:

#### Tiny-imagenet dataset:
- download the dataset [tiny-imagenet-200.zip](https://cs231n.stanford.edu/tiny-imagenet-200.zip) into dir `./utils` 
- reformat the dataset.
```
cd ./utils
./process_tiny_data.sh
```

#### GTRSB dataset:
download the dataset [gtrsb.zip](https://benchmark.ini.rub.de/gtrsb_dataset.html)
- reformat the dataset.
select GTRSB when loading the data set to format it.


#### Others:
CIFAR will be automatically download


### Reproduce experiments: 

- run experiments for the four datasets:
```
python main.py --params utils/X.yaml
```
`X` = `cifar_params`or`tiny_params`  ``. Parameters can be changed in those yaml files to reproduce our experiments.




## Citation
If you find our work useful in your research, please consider citing:

Guorui Li, Runxing Chang, Ying Wang, Cong Wang. Dual-domain based backdoor attack against federated learning. Neurocomputing, 2025, 129424: 1-13.





