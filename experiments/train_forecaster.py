import os
import sys
import json

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from experiments.utils import dataset
from utils.data_augmentation import DataAugmentation
from utils.latent_dataset import compute_latent_dataset
from torch.utils.data import DataLoader
from experiments.utils.model_building import build_forecaster, build_and_load_trained_model
from utils import training

from escnn import gspaces

from argparse import ArgumentParser, ArgumentTypeError

########################
# Parsing arguments
########################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()

# script parameters
parser.add_argument('model', type=str, choices=['3DCNN', 'CNN', 'steerable3DCNN', 'steerableCNN'])
parser.add_argument('train_name', type=str)
parser.add_argument('ae_model_name', type=str)
parser.add_argument('ae_train_name', type=str)
parser.add_argument('epochs', type=int)
parser.add_argument('-start_epoch', type=int, default=-1)
parser.add_argument('-only_save_best', type=str2bool, default=True)
parser.add_argument('-train_loss_in_eval', action='store_true', default=False)

# data parameters
parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300')
parser.add_argument('-n_train', type=int, default=-1)
parser.add_argument('-n_valid', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=64)

# model hyperparameters
parser.add_argument('-flips', type=str2bool, default=True)
parser.add_argument('-rots', type=int, default=4)
parser.add_argument('-v_kernel_size', type=int, default=3)
parser.add_argument('-h_kernel_size', type=int, default=3)
parser.add_argument('-drop_rate', type=float, default=0.2)
parser.add_argument('-recurrent_drop_rate', type=float, default=0)
parser.add_argument('-nonlinearity', type=str, default='tanh', choices=['tanh', 'ReLU', 'ELU'])
parser.add_argument('-lstm_channels', nargs='+', type=int, default=None)
parser.add_argument('-weight_decay', type=float, default=0)
parser.add_argument('-residual_connection', type=str2bool, default=True)
parser.add_argument('-use_lstm_encoder', type=str2bool, default=True)

# training hyperparameters
parser.add_argument('-warmup_seq_length', type=int, default=25)
parser.add_argument('-forecast_seq_length', type=int, default=50)
parser.add_argument('-parallel_ops', action='store_true', default=False)
parser.add_argument('-loss_on_decoded', action='store_true', default=False)
parser.add_argument('-init_forced_decoding_prob', type=float, default=1)
parser.add_argument('-min_forced_decoding_prob', type=float, default=0)
parser.add_argument('-forced_decoding_epochs', type=int, default=100)
parser.add_argument('-use_force_decoding', type=str2bool, default=True)
parser.add_argument('-backprop_through_autoregression', type=str2bool, default=True)

parser.add_argument('-train_autoencoder', action='store_true', default=False)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-no_lr_scheduler', dest='use_lr_scheduler', action='store_false', default=True)
parser.add_argument('-lr_decay', type=float, default=0.5)
parser.add_argument('-lr_decay_patience', type=int, default=5)
parser.add_argument('-early_stopping', type=int, default=20)
parser.add_argument('-early_stopping_threshold', type=float, default=1e-5)

args = parser.parse_args()

if args.train_autoencoder and not args.loss_on_decoded:
    raise Exception('When training the autoencoder, the loss must be computed on the decoded output.')

if args.loss_on_decoded and args.use_force_decoding:
    raise Exception('When training on decoded data, force decoding the latent space is not supported')

if not args.use_force_decoding:
    args.init_forced_decoding_prob = 0
    args.min_forced_decoding_prob = 0

########################
# Seed and GPU
########################
import torch, numpy as np, random
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(0)
    DEVICE = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(DEVICE))
else:
    print('Failed to find GPU. Will use CPU.')
    DEVICE = 'cpu'
    
    
models_dir = os.path.join(EXPERIMENT_DIR, 'trained_models')
########################
# Data
########################
BATCH_SIZE = args.batch_size

SIMULATION_NAME = args.simulation_name

sim_file = os.path.join(EXPERIMENT_DIR, '..', 'data', 'datasets', f'{SIMULATION_NAME}.h5')


if not args.loss_on_decoded:
    latent_file = os.path.join(EXPERIMENT_DIR, 'latent_datasets', args.ae_model_name, 
                               args.ae_train_name, f'{SIMULATION_NAME}.h5')
    if not os.path.isfile(latent_file):
        print('Loading autoencoder to precompute latent dataset')
        autoencoder = build_and_load_trained_model(models_dir, os.path.join('AE', args.ae_model_name), args.ae_train_name)
        autoencoder.to(DEVICE)
        print('Precompute latent dataset')
        
        # data augmentation is already included in the dataset itself since transformation laws do not apply
        # on latent space for non-equivariant models
        latent_data_aug = DataAugmentation(in_height=autoencoder.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))
        
        compute_latent_dataset(autoencoder, latent_file, sim_file, device=DEVICE, batch_size=args.batch_size,
                               data_augmentation=latent_data_aug)
        del autoencoder
    sim_file = latent_file

train_dataset = dataset.RBForecastDataset(sim_file, 'train', device=DEVICE, shuffle=True, samples=args.n_train, warmup_seq_length=args.warmup_seq_length, 
                                          forecast_seq_length=args.forecast_seq_length, data_aug=True)
valid_dataset = dataset.RBForecastDataset(sim_file, 'valid', device=DEVICE, shuffle=True, samples=args.n_valid, warmup_seq_length=args.warmup_seq_length, 
                                          forecast_seq_length=args.forecast_seq_length)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)



########################
# Hyperparameter
########################

H_KERNEL_SIZE, V_KERNEL_SIZE = args.h_kernel_size, args.v_kernel_size
DROP_RATE = args.drop_rate
RECURRENT_DROP_RATE = args.recurrent_drop_rate
NONLINEARITY = args.nonlinearity

LEARNING_RATE = args.lr
LR_DECAY = args.lr_decay
LR_DECAY_PATIENCE = args.lr_decay_patience # epochs at which the learning rate is multiplied by LR_DECAY
USE_LR_SCHEDULER = args.use_lr_scheduler
WEIGHT_DECAY = args.weight_decay
EARLY_STOPPING = args.early_stopping # early stopping patience
EARLY_STOPPING_THRESHOLD = args.early_stopping_threshold # early stopping patience

OPTIMIZER = torch.optim.Adam

########################
# Building Model
########################
print('Building model...')


FLIPS, ROTS = args.flips, args.rots
match args.model:
    case 'steerableCNN':
        print(f'Selected Steerable CNN with {ROTS=}, {FLIPS=}')
        lstm_channels = {
            (True, 4): (32, 32)
            }[(FLIPS, ROTS)]
    case 'steerable3DCNN':
        print(f'Selected Steerable 3D CNN with {ROTS=}, {FLIPS=}')
        lstm_channels = {
            (True, 4): (32, 32)
            }[(FLIPS, ROTS)]
    case 'CNN':
        print('Selected CNN')
        lstm_channels = (32, 32)
    case '3DCNN':
        print('Selected 3DCNN')
        lstm_channels = (32, 32)
        
if args.lstm_channels is not None:
    lstm_channels = args.lstm_channels
    

model_hyperparameters = {
    'model_type': args.model,
    'ae_model_name': args.ae_model_name,
    'ae_train_name': args.ae_train_name,
    'simulation_name': SIMULATION_NAME,
    'h_kernel_size': H_KERNEL_SIZE,
    'v_kernel_size': V_KERNEL_SIZE,
    'drop_rate': DROP_RATE,
    'recurrent_drop_rate': RECURRENT_DROP_RATE,
    'nonlinearity': NONLINEARITY,
    'flips': FLIPS,
    'rots': ROTS,
    'lstm_channels': lstm_channels,
    'parallel_ops': args.parallel_ops,
    'residual_connection': args.residual_connection,
    'train_autoencoder': args.train_autoencoder,
    'include_autoencoder': args.loss_on_decoded,
    'use_lstm_encoder': args.use_lstm_encoder,
    'init_forced_decoding_prob': args.init_forced_decoding_prob,
    'min_forced_decoding_prob': args.min_forced_decoding_prob,
    'forced_decoding_epochs': args.forced_decoding_epochs,
    'use_force_decoding': args.use_force_decoding,
    'backprop_through_autoregression': args.backprop_through_autoregression
}

model = build_forecaster(models_dir=models_dir, **model_hyperparameters)

model.to(DEVICE)
model.summary()

########################
# Prepare Training
########################

model_name = {'steerableCNN': f'{"D" if FLIPS else "C"}{ROTS}cnn',
              'steerable3DCNN': f'3D-{"D" if FLIPS else "C"}{ROTS}cnn',
              'CNN': 'cnn',
              '3DCNN': '3Dcnn'}[args.model]
model_name = os.path.join('FC', model_name)
train_name = args.train_name
train_dir = os.path.join(models_dir, model_name, train_name)
os.makedirs(train_dir, exist_ok=True)

loss_fn = torch.nn.MSELoss()

trainable_parameters = model.parameters()
optimizer = OPTIMIZER(trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

if args.loss_on_decoded:
    data_augmentation = DataAugmentation(in_height=model.in_height, gspace=gspaces.flipRot2dOnR2(N=4))
else:
    # data augmentation is already included in the latent dataset
    data_augmentation = None


START_EPOCH = args.start_epoch # loads pretrained model if greater 0, loads last available epoch for -1

initial_early_stop_count, loaded_epoch = training.load_trained_model(model=model, 
                                                                     optimizer=optimizer, 
                                                                     models_dir=models_dir, 
                                                                     model_name=model_name, 
                                                                     train_name=train_name,
                                                                     epoch=START_EPOCH)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_DECAY_PATIENCE, 
                                                          factor=LR_DECAY, threshold=1e-5)

EPOCHS = args.epochs - loaded_epoch


########################
# Save Hyperparameters
########################

train_hyperparameters = {
    'batch_size': BATCH_SIZE,
    'n_train': train_dataset.num_samples,
    'n_valid': valid_dataset.num_samples,
    'learning_rate': LEARNING_RATE,
    'optimizer': str(OPTIMIZER),
    'lr_decay': LR_DECAY,
    'lr_decay_patience': LR_DECAY_PATIENCE,
    'use_lr_scheduler': USE_LR_SCHEDULER,
    'weight_decay': WEIGHT_DECAY,
    'early_stopping': EARLY_STOPPING,
    'early_stopping_threshold': EARLY_STOPPING_THRESHOLD,
    'epochs': loaded_epoch+EPOCHS,
    'train_loss_in_eval': args.train_loss_in_eval,
    'warmup_seq_length': args.warmup_seq_length,
    'forecast_seq_length': args.forecast_seq_length,
}

hyperparameters = model_hyperparameters | train_hyperparameters

hyperparameters['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

hp_file = os.path.join(train_dir, 'hyperparameters.json')
if loaded_epoch > 0 and os.path.isfile(hp_file):
    with open(hp_file, 'r') as f:
        prev_hps = json.load(f)
        prev_hps['epochs'] = hyperparameters['epochs'] # ignore epochs
        assert hyperparameters == prev_hps, f"New hyperparameters do not correspond to the old ones"
else:
    with open(hp_file, 'w+') as f:
        json.dump(hyperparameters, f, indent=4)
        


########################
# Training
########################
if args.loss_on_decoded:
    if args.train_autoencoder:
        for param in model.autoencoder.parameters():
            param.requires_grad = True
    else:
        # Freeze the autoencoder parameters so they are not updated during training.
        for param in model.autoencoder.parameters():
            param.requires_grad = False
    

training.train(model=model, models_dir=models_dir, model_name=model_name, train_name=train_name, start_epoch=loaded_epoch, 
               epochs=EPOCHS, train_loader=train_loader, valid_loader=valid_loader, loss_fn=loss_fn, optimizer=optimizer, 
               lr_scheduler=lr_scheduler, use_lr_scheduler=USE_LR_SCHEDULER, early_stopping=EARLY_STOPPING, only_save_best=args.only_save_best, 
               train_samples=train_dataset.num_samples, batch_size=BATCH_SIZE, data_augmentation=data_augmentation, plot=False, 
               initial_early_stop_count=initial_early_stop_count, train_loss_in_eval=args.train_loss_in_eval, early_stopping_threshold=EARLY_STOPPING_THRESHOLD, 
               model_forward_kwargs={'steps': args.forecast_seq_length, 'epoch': None, 'ground_truth': None})