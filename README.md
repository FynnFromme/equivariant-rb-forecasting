# Surrogate Modeling of 3D Rayleigh-Bénard Convection with Equivariant Autoencoders

This repository is the official implementation of the paper ["Surrogate Modeling of 3D Rayleigh-Bénard Convection with Equivariant Autoencoders"](https://arxiv.org/abs/2505.13569). 

![forecast](https://github.com/user-attachments/assets/2189e52a-c1fd-4775-b2a7-c3ef71627b6c)

## Project Structure
In the following, an overview of the most important parts of the project is shown:
```
├── data                  | datasets used for training and evaluation
├── experiments          
│   ├── models           
│   │   ├── autoencoder   | implementation of autoencoder models
│   │   └── forecasters   | implementation of forecasting models
│   ├── results           | evaluation results
│   ├── trained_models    | trained models, their HPs and training logs
│   └── experiments.ipynb | visualization of results
├── layers
│   ├── conv              | implementation of convolutional layers
│   └── lstm              | implementation of convolutional LSTMs
└── simulation            | simulation of 3D Rayleigh-Bénard
```

## Requirements

To install the requirements, run:

```setup
conda env create -f environment.yml
```


## 3D Rayleigh-Bénard Simulation
For details on how to simulate 3D Rayleigh-Bénard see the help page of the corresponding script:
```
julia --project=simulation simulation/3d/RayleighBenard3D.jl --help
```

## Data Preparation
For details on how to create standardized train, validation and test datasets of 3D Rayleigh-Bénard simulations, see the help page of the corresponding script:
```
python data/data_preparation.py --help
```

## Training

To train a steerable **autoencoder** for 100 epochs with the name "ae1", run the following command:

```train
python experiments/train_autoencoder.py SteerableConv ae1 100
```

To train a steerable **forecaster** using the previously trained D4cnn autoencoder for 100 epochs, run this command:

```train
python experiments/train_forecaster.py SteerableConv fc1 D4cnn ae1 100
```

For more details on all possible arguments see: 
```
python experiments/train_autoencoder.py --help
python experiments/train_forecaster.py --help
```

## Evaluation

To evaluate an **autoencoder**, run the following command with a selection of the given options:

```eval
python experiments/evaluate.py AE/D4cnn ae1 -eval_performance
```

To evaluate an **forecaster**, run the following command with a selection of the given options:

```eval
python experiments/evaluate.py FC/D4cnn fc1 -eval_autoregressive_performance
```

For more details on all possible arguments see: 
```
python experiments/evaluate.py --help
```


<!-- ## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.