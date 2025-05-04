import os
import sys
import json

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from experiments.utils import dataset
from torch.utils.data import DataLoader
from utils.data_augmentation import DataAugmentation
from experiments.utils.model_building import build_forecaster
from utils import training

from escnn import gspaces
import torch, numpy as np, random

from argparse import ArgumentParser, Namespace
from utils.argument_parsing import str2bool


TRAINED_MODELS_DIR = os.path.join(EXPERIMENT_DIR, 'trained_models')
DATA_DIR = os.path.join(EXPERIMENT_DIR, '..', 'data')


########################
# Parsing arguments
########################
parser = ArgumentParser()

parser.add_argument('model_name', type=str,
                    help='The model name of the trained forecaster (e.g. cnn, 3Dcnn, D4cnn, 3D-D4cnn)')
parser.add_argument('fc_train_name', type=str,
                    help='The name of the trained forecaster.')
parser.add_argument('epochs', type=int,
                    help='The number of epochs to train the model.')
parser.add_argument('-start_epoch', type=int, default=-1,
                    help='When restarting training, start_epoch specifies the saved epoch used as a \
                        starting point. For `-1`, the latest saved epoch will be used. Set to `0` to not \
                        load a previous state. Defaults to `-1`.')
parser.add_argument('-only_save_best', type=str2bool, default=False,
                    help='When set to `True`, previously saved epochs will be deleted once a better \
                        validation accuracy is achieved. Defaults to `True`.')
parser.add_argument('-train_loss_in_eval', type=str2bool, default=False,
                    help='When set to `True`, the training loss will be computed another time in eval mode \
                        and therefore mitigating the effect of e.g. Dropout. Note that this slows down training. \
                        Defaults to `False`.')

# data parameters
parser.add_argument('-dataset_dir', type=str, default=os.path.join(DATA_DIR, 'datasets'),
                    help=f'The directory of the dataset. Defaults to "data/datasets".')
parser.add_argument('-n_train', type=int, default=-1,
                    help='The number of samples used for training. Set to `-1` to use all available samples. \
                        Defaults to `-1`.')
parser.add_argument('-n_valid', type=int, default=-1,
                    help='The number of samples used for validation. Set to `-1` to use all available samples. \
                        Defaults to `-1`.')
parser.add_argument('-batch_size', type=int, default=64,
                    help='The batch size used during training. Defaults to `64`.')
parser.add_argument('-accumulated_batches', type=int, default=1,
                    help='The number of batches the loss is accumulated before performing a gradient descent step. \
                        Defaults to `1`.')


# training hyperparameters
parser.add_argument('-warmup_seq_length', type=int, default=25,
                    help='The length of the warmup sequence, which the model gets as input during training \
                        Defaults to `25`.')
parser.add_argument('-forecast_seq_length', type=int, default=50,
                    help='The length of the forecasted sequence during training. Defaults to `50`.')
parser.add_argument('-parallel_ops', type=str2bool, default=False,
                    help='Whether to apply certain operations (like autoencoder, output head, etc.) \
                        in parallel to the whole sequence. Defaults to `False`.')
parser.add_argument('-backprop_through_autoregression', type=str2bool, default=True,
                    help='Whether to backpropagate through autoregressive steps. Defaults to `True`.')

parser.add_argument('-lr', type=float, default=1e-3,
                    help='The learning rate. Defaults to `1e-3`.')
parser.add_argument('-use_lr_scheduler', type=str2bool, default=True,
                    help='Whether to use the `ReduceLROnPlateau` learning rate scheduler. Defaults to `True`.')
parser.add_argument('-lr_decay', type=float, default=0.5,
                    help='The factor applied to the learning rate when the validation loss stagnates. \
                        Defaults to `0.5`.')
parser.add_argument('-lr_decay_patience', type=int, default=5,
                    help='The number of epochs without an improvement in validation performance until \
                        the learning rate decays. Defaults to `5`.')
parser.add_argument('-early_stopping', type=int, default=20,
                    help='The number of epochs without an improvement in validation performance until \
                        training is stopped. Defaults to `20`.')
parser.add_argument('-early_stopping_threshold', type=float, default=1e-5,
                    help='The threshold which must be surpassed in order to count as an improvement in \
                        validation performance in the context of early stopping. Defaults to `1e-5`.')
parser.add_argument('-seed', type=int, default=0,
                    help='The seed used for initializing the model. Defaults to `0`.')

args = parser.parse_args()
args.model_name = 'FC/' + args.model_name

########################
# Forecaster Hyperparameters
########################
fc_train_dir = os.path.join(TRAINED_MODELS_DIR, args.model_name, args.fc_train_name)

fc_hp_file = os.path.join(fc_train_dir, 'hyperparameters.json')
with open(fc_hp_file, 'r') as f:
    fc_hps = json.load(f)
    
args = Namespace(**(fc_hps | vars(args)))

args.train_name = args.fc_train_name + '_finetuned'
    

########################
# Seed and GPU
########################
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
        
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(DEVICE))
else:
    print('Failed to find GPU. Will use CPU.')
    DEVICE = 'cpu'
    
    
########################
# Data
########################
sim_file = os.path.join(args.dataset_dir, f'{args.simulation_name}.h5')

train_dataset = dataset.RBForecastDataset(sim_file, 'train', device=DEVICE, shuffle=True, samples=args.n_train, 
                                          warmup_seq_length=args.warmup_seq_length, forecast_seq_length=args.forecast_seq_length)
valid_dataset = dataset.RBForecastDataset(sim_file, 'valid', device=DEVICE, shuffle=True, samples=args.n_valid, 
                                          warmup_seq_length=args.warmup_seq_length, forecast_seq_length=args.forecast_seq_length)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)


########################
# Building Model
########################
# select default encoder channels if not specified   

model_hyperparameters = {
    'parallel_ops': args.parallel_ops,
    'train_autoencoder': True,
    'include_autoencoder': True,
    'backprop_through_autoregression': args.backprop_through_autoregression,
    'init_forced_decoding_prob': 0,
    'min_forced_decoding_prob': 0,
    'forced_decoding_epochs': 0,
    'use_forced_decoding': False
}

model_hyperparameters = fc_hps | model_hyperparameters

print('Building model...')
model = build_forecaster(models_dir=TRAINED_MODELS_DIR, **model_hyperparameters)
model.to(DEVICE)

model.summary()


########################
# Prepare Training
########################
train_dir = os.path.join(TRAINED_MODELS_DIR, args.model_name, args.train_name)
os.makedirs(train_dir, exist_ok=True)

loss_fn = torch.nn.MSELoss()
trainable_parameters = model.parameters()
optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_decay_patience, 
                                                          factor=args.lr_decay, threshold=1e-5)

# data augmentation only by 90° rotations for efficiency reasons
data_augmentation = DataAugmentation(in_height=model.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))

# loads pretrained model if start_epoch > 0, loads last available epoch for -1
initial_early_stop_count, loaded_epoch = training.load_trained_model(model=model, 
                                                                     optimizer=optimizer, 
                                                                     models_dir=TRAINED_MODELS_DIR, 
                                                                     model_name=args.model_name, 
                                                                     train_name=args.train_name,
                                                                     epoch=args.start_epoch)

if loaded_epoch == 0:
    # load non fine-tuned forecaster and autoencoder
    _, fc_loaded_epoch = training.load_trained_model(model=model.latent_forecaster, 
                                                     models_dir=TRAINED_MODELS_DIR, 
                                                     model_name=args.model_name, 
                                                     train_name=args.fc_train_name, 
                                                     epoch=-1)
    
    _, ae_loaded_epoch = training.load_trained_model(model=model.autoencoder, 
                                                     models_dir=TRAINED_MODELS_DIR, 
                                                     model_name='AE/'+args.ae_model_name, 
                                                     train_name=args.ae_train_name,
                                                     epoch=-1)
    
    assert ae_loaded_epoch > 0, 'Trained autoencoder not found'
    assert fc_loaded_epoch > 0, 'Trained forecaster not found'

remaining_epochs = args.epochs - loaded_epoch


########################
# Save Hyperparameters
########################
train_hyperparameters = {
    'batch_size': args.batch_size,
    'accumulated_batches': args.accumulated_batches,
    'n_train': train_dataset.num_samples,
    'n_valid': valid_dataset.num_samples,
    'learning_rate': args.lr,
    'optimizer': str(optimizer.__class__),
    'lr_decay': args.lr_decay,
    'lr_decay_patience': args.lr_decay_patience,
    'use_lr_scheduler': args.use_lr_scheduler,
    'early_stopping': args.early_stopping,
    'early_stopping_threshold': args.early_stopping_threshold,
    'epochs': args.epochs,
    'train_loss_in_eval': args.train_loss_in_eval,
    'warmup_seq_length': args.warmup_seq_length,
    'forecast_seq_length': args.forecast_seq_length,
}

hyperparameters = model_hyperparameters | train_hyperparameters

hyperparameters['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

hp_file = os.path.join(train_dir, 'hyperparameters.json')
if loaded_epoch > 0 and os.path.isfile(hp_file):
    # check whether new hyperparameters are the same as the ones from the loaded model
    with open(hp_file, 'r') as f:
        prev_hps = json.load(f)
        prev_hps['epochs'] = hyperparameters['epochs'] # ignore epochs
        assert hyperparameters == prev_hps, f"New hyperparameters do not correspond to the old ones"
        
# save hyperparameters
with open(hp_file, 'w+') as f:
    json.dump(hyperparameters, f, indent=4)


########################
# Training
########################
for param in model.autoencoder.parameters():
    param.requires_grad = True
    
# epoch and ground_truth will be replaced by training script
model_forward_kwargs = {'steps': args.forecast_seq_length}
training.train(model=model, models_dir=TRAINED_MODELS_DIR, model_name=args.model_name, train_name=args.train_name, 
               start_epoch=loaded_epoch, epochs=remaining_epochs, train_loader=train_loader, valid_loader=valid_loader, 
               loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=args.use_lr_scheduler, 
               early_stopping=args.early_stopping, only_save_best=args.only_save_best, 
               train_samples=train_dataset.num_samples, model_forward_kwargs=model_forward_kwargs, 
               initial_early_stop_count=initial_early_stop_count, train_loss_in_eval=args.train_loss_in_eval, 
               early_stopping_threshold=args.early_stopping_threshold, batch_size=args.batch_size, 
               accumulated_batches=args.accumulated_batches, data_augmentation=data_augmentation, plot=False)