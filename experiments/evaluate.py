import os
import sys
import time
import json
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from utils import dataset
from utils.latent_dataset import compute_latent_dataset
from torch.utils.data import DataLoader

from utils.evaluation import compute_loss, compute_loss_per_channel, compute_autoregressive_loss, compute_predictions
from utils import visualization
from utils.evaluation import compute_latent_sensitivity
from utils.model_building import build_and_load_trained_model

import escnn
import torch, numpy as np, random

from argparse import ArgumentParser

from tqdm.auto import tqdm


TRAINED_MODELS_DIR = os.path.join(EXPERIMENT_DIR, 'trained_models')
DATA_DIR = os.path.join(EXPERIMENT_DIR, '..', 'data')


########################
# Parsing arguments
########################
parser = ArgumentParser()

parser.add_argument('model_name', type=str,
                    help='The model name (e.g. 3Dcnn, D4cnn) of the model to evaluate together \
                    with "AE/" or "FC/" as a prefix to sepcify whether the model is an autoencoder or forecaster.')
parser.add_argument('train_name', type=str,
                    help='The name of the trained model to evaluate.')

parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300',
                    help='The name of the dataset used to train the model. It must be located either in "data/datasets" \
                        or otherwise in the directory specified by `-dataset_dir`. \
                        Defaults to "x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300".')
parser.add_argument('-dataset_dir', type=str, default=os.path.join(DATA_DIR, 'datasets'),
                    help=f'The directory of the dataset. Defaults to "data/datasets".')
parser.add_argument('-latent_dataset_dir', type=str, default=os.path.join(EXPERIMENT_DIR, 'latent_datasets'),
                    help=f'The directory of the latent-dataset. Defaults to "experiments/latent_datasets".')
parser.add_argument('-n_test', type=int, default=-1,
                    help='The number of samples used for evaluating the performance on the test set. Set \
                        to `-1` to use all available samples. Defaults to `-1`.')
parser.add_argument('-n_train', type=int, default=-1,
                    help='The number of samples used for evaluating the performance on the training set. Set \
                        to `-1` to use all available samples. Defaults to `-1`.')
parser.add_argument('-batch_size', type=int, default=64,
                    help='The batch size used during evaluation. Defaults to `64`.')
parser.add_argument('-seed', type=int, default=0,
                    help='The seed used for evaluation. Defaults to `0`.')

parser.add_argument('-eval_performance', action='store_true', default=False,
                    help='When given, the performance of the model is evaluated.')
parser.add_argument('-eval_autoregressive_performance', action='store_true', default=False,
                    help='When given, the autoregressive performance of a forecaster is evaluated.')
parser.add_argument('-eval_performance_per_sim', action='store_true', default=False,
                    help='When given, the performance of the model is evaluated per simulation.')
parser.add_argument('-eval_performance_per_channel', action='store_true', default=False,
                    help='When given, the performance of the model is evaluated per channel.')
parser.add_argument('-eval_performance_per_height', action='store_true', default=False,
                    help='When given, the performance of the model is evaluated per height.')
parser.add_argument('-eval_inference_time', action='store_true', default=False,
                    help='When given, the average inverence time is evaluated.')
parser.add_argument('-check_equivariance', action='store_true', default=False,
                    help='When given, the equivariance of an equivariant model is checked.')
parser.add_argument('-animate2d', action='store_true', default=False,
                    help='When given, the output of the model is animated in 2D.')
parser.add_argument('-animate3d', action='store_true', default=False,
                    help='When given, the output of the model is animated in 3D.')
parser.add_argument('-compute_latent_sensitivity', action='store_true', default=False,
                    help='When given, the patterns represented by the entries in latent space are calculated.')

parser.add_argument('-autoregressive_confidence_interval', type=float, default=0.95,
                    help='The confidence inteval used to compute the uncertainty bounds for \
                        autoregressive performance of a forecaster. Defaults to `0.95`.')
parser.add_argument('-animation_samples', type=int, default=np.inf,
                    help='The number of snapshots to visualize. Defaults to all available.')
parser.add_argument('-latent_sensitivity_samples', type=int, default=-1,
                    help='The number of snapshots to compute the latent patterns. Defaults to all available.')
 # the number of channels in latent space to compute sensitivty in parallel for
parser.add_argument('-parallel_channels_latent_sensitivity', type=int, default=1,
                    help='The number of latent channels to compute the patterns in parallel for. Defaults to `1`.')

parser.add_argument('-loss_on_latent', action='store_true', default=False,
                    help='When given, the performance of a forecaster is evaluated in the latent space.')
parser.add_argument('-warmup_seq_length', type=int, default=25,
                    help='The length of the warmup sequence, which the model gets as input during evaluation \
                        Defaults to `25`.')
parser.add_argument('-forecast_seq_length', type=int, default=50,
                    help='The length of the forecasted sequence during evaluating performance. Defaults to `50`.')
parser.add_argument('-autoregressive_warmup_seq_length', type=int, default=50,
                    help='The length of the warmup sequence, which the model gets as input during \
                        *autoregressive* evaluation. Defaults to `25`.')
parser.add_argument('-autoregressive_forecast_seq_length', type=int, default=100,
                    help='The length of the forecasted sequence during evaluating *autoregressive* performance. \
                        Defaults to `50`.')

args = parser.parse_args()

if not any([args.eval_performance, 
            args.eval_autoregressive_performance,
            args.eval_performance_per_sim,
            args.eval_performance_per_channel,
            args.eval_performance_per_height,
            args.eval_inference_time,
            args.check_equivariance, 
            args.animate2d, 
            args.animate3d, 
            args.compute_latent_sensitivity]):
    parser.error("""
                 No evaluation method selected.
                 Please add at least one of the following flags:
                 -eval_performance
                 -eval_autoregressive_performance
                 -eval_performance_per_sim
                 -eval_performance_per_channel
                 -eval_performance_per_height
                 -eval_inference_time
                 -check_equivariance
                 -animate2d
                 -animate3d
                 -compute_latent_sensitivity
                 """)

is_forecaster = args.model_name.startswith('FC')
is_autoencoder = args.model_name.startswith('AE')

if not is_forecaster and not is_autoencoder:
    raise parser.error('The model type (autoencoder or forecaster) must be specified in model_name : \
        "AE/<model_name>" or "FC/<model_name>"')


########################
# Seed and GPU
########################
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
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
anim_sim_file = os.path.join(args.dataset_dir, f'{args.simulation_name}.h5')

if is_autoencoder:
    test_dataset = dataset.RBDataset(sim_file, 'test', device=DEVICE, shuffle=False, samples=args.n_test)
    if args.n_train != 0:
        train_dataset = dataset.RBDataset(sim_file, 'train', device=DEVICE, shuffle=False, samples=args.n_train)
else:
    if args.loss_on_latent: # requires latent representations which are precomputed
        # determine autoencoder of forecaster
        hp_file = os.path.join(TRAINED_MODELS_DIR, args.model_name, args.train_name, 'hyperparameters.json')
        with open(hp_file, 'r') as f:
            hps = json.load(f)
            ae_model_name = hps['ae_model_name']
            ae_train_name = hps['ae_train_name']
        
        latent_file = os.path.join(args.latent_dataset_dir, ae_model_name, ae_train_name, f'{args.simulation_name}.h5')
        if not os.path.isfile(latent_file):
            print('Loading autoencoder to precompute latent dataset...')
            autoencoder = build_and_load_trained_model(TRAINED_MODELS_DIR, os.path.join('AE', ae_model_name), ae_train_name)
            autoencoder.to(DEVICE)
            
            print('Precompute latent dataset...')
            compute_latent_dataset(autoencoder, latent_file, sim_file, device=DEVICE, batch_size=args.batch_size)
            
            del autoencoder
        sim_file = latent_file
        
    test_dataset = dataset.RBForecastDataset(sim_file, 'test', device=DEVICE, shuffle=False, samples=args.n_test, 
                                             forecast_seq_length=args.forecast_seq_length,
                                             warmup_seq_length=args.warmup_seq_length)
    if args.n_train != 0:
        train_dataset = dataset.RBForecastDataset(sim_file, 'train', device=DEVICE, shuffle=False, samples=args.n_train, 
                                                  forecast_seq_length=args.forecast_seq_length,
                                                  warmup_seq_length=args.warmup_seq_length)
    
    autoregressive_test_dataset = dataset.RBForecastDataset(
        sim_file, 'test', device=DEVICE, shuffle=False, samples=args.n_test, 
        warmup_seq_length=args.autoregressive_warmup_seq_length, 
        forecast_seq_length=args.autoregressive_forecast_seq_length,
        y_to_device=False)
    autoregressive_test_loader = DataLoader(autoregressive_test_dataset, batch_size=args.batch_size)
    if args.n_train != 0:
        autoregressive_train_dataset = dataset.RBForecastDataset(
            sim_file, 'train', device=DEVICE, shuffle=False,
            samples=args.n_train, warmup_seq_length=args.autoregressive_warmup_seq_length, 
            forecast_seq_length=args.autoregressive_forecast_seq_length,
            y_to_device=False)
        autoregressive_train_loader = DataLoader(autoregressive_train_dataset, batch_size=args.batch_size)
    
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
if args.n_train != 0:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)


########################
# Load Model
########################
is_forecast_model = args.model_name.startswith('FC')

override_hps = {'include_autoencoder': not args.loss_on_latent, 'parallel_ops': False} if is_forecast_model else {}
model = build_and_load_trained_model(TRAINED_MODELS_DIR, args.model_name, args.train_name, epoch=-1, override_hps=override_hps)
model.to(DEVICE) 

model_forward_kwargs = {'steps': args.forecast_seq_length} if is_forecast_model else {}
   
   
########################
# Evaluate Model
########################  
def load_json(json_file):
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        return {}   
    
def save_json(dict, json_file):
    with open(json_file, 'w+') as f:
        json.dump(dict, f, indent=4)

results_dir = os.path.join(EXPERIMENT_DIR, 'results', args.model_name, args.train_name)
os.makedirs(results_dir, exist_ok=True)


if args.eval_performance:
    print('Evaluating model performance...')
    # read current performances
    performance_file = os.path.join(results_dir, 'performance.json')
    performance = load_json(performance_file)
    performance['simulation_name'] = args.simulation_name
    
    # update new performance metrics but keep other contents
    mse, mae = compute_loss(model, test_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                            test_dataset.num_samples, args.batch_size, model_forward_kwargs)
    performance['mse'], performance['mae'] = mse, mae
    performance['rmse'] = np.sqrt(mse)
    
    print(f'MSE={performance["mse"]:.4f}')
    print(f'RMSE={performance["rmse"]:.4f}')
    print(f'MAE={performance["mae"]:.4f}')
    
    if args.n_train != 0:
        mse_train, mae_train = compute_loss(model, train_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                            train_dataset.num_samples, args.batch_size, model_forward_kwargs)
        performance['mse_train'], performance['mae_train'] = mse_train, mae_train
        performance['rmse_train'] = np.sqrt(mse_train)

        print(f'MSE_train={performance["mse_train"]:.4f}')
        print(f'RMSE_train={performance["rmse_train"]:.4f}')
        print(f'MAE_train={performance["mae_train"]:.4f}')
    
    # update performances in file
    save_json(performance, performance_file)
    
    
if args.eval_inference_time:
    print('Evaluating inference time...')
    # read current times
    inference_time_file = os.path.join(results_dir, 'inference_time.json')
    inference_time = load_json(inference_time_file)
    inference_time['batch_size'] = args.batch_size
    
    if is_forecaster:
        inference_time['loss_on_latent'] = args.loss_on_latent
        inference_time['forecast_seq_length'] = args.forecast_seq_length
        inference_time['warmup_seq_length'] = args.warmup_seq_length
    
    predictions = compute_predictions(model, test_loader, test_dataset.num_samples, args.batch_size, model_forward_kwargs)
    
    inference_times = []
    t = time.time()
    for outputs, predictions in predictions:
        delta_t = time.time()-t
        if predictions.size(0) == args.batch_size: # ignore last (incomplete) batch
            inference_times.append(delta_t)
        t = time.time()
        
    inference_time['avg_time'] = np.mean(inference_times)
    
    # update inference_time in file
    save_json(inference_time, inference_time_file)
        
        
if args.eval_autoregressive_performance:
    print('Computing autoregressive performance...')
    if args.model_name.startswith('AE'):
        print('Autoregressive performance can only be computed for forecasters')
    else:
        # read current performances
        performance_file = os.path.join(results_dir, 'autoregressive_performance.json')
        performance = load_json(performance_file)
        performance['simulation_name'] = args.simulation_name

        # update new performance metrics but keep other contents
        avgs, medians, lower_bounds, upper_bounds, std = compute_autoregressive_loss(
            model, args.autoregressive_forecast_seq_length, autoregressive_test_loader,
            [torch.nn.MSELoss(), torch.nn.L1Loss()], args.autoregressive_forecast_seq_length,
            autoregressive_test_dataset.num_samples, args.batch_size, args.autoregressive_confidence_interval)
        
        performance['mse'], performance['mae'] = avgs
        performance['mse_median'], performance['mae_median'] = medians
        performance['mse_lower'], performance['mae_lower'] = lower_bounds
        performance['mse_upper'], performance['mae_upper'] = upper_bounds
        performance['mse_std'], performance['mae_std'] = std
        performance['rmse'] = list(np.sqrt(performance['mse']))
        performance['rmse_median'] = list(np.sqrt(performance['mse_median']))
        performance['rmse_lower'] = list(np.sqrt(performance['mse_lower']))
        performance['rmse_upper'] = list(np.sqrt(performance['mse_upper']))
        performance['rmse_std'] = list(np.sqrt(performance['mse_std']))
        
        # update performances in file
        save_json(performance, performance_file)
        
        if args.n_train != 0:
            avgs, medians, lower_bounds, upper_bounds, std = compute_autoregressive_loss(
                model, args.autoregressive_forecast_seq_length, autoregressive_train_loader,
                [torch.nn.MSELoss(), torch.nn.L1Loss()], args.autoregressive_forecast_seq_length,
                autoregressive_train_dataset.num_samples, args.batch_size, args.autoregressive_confidence_interval)
            
            performance['mse_train'], performance['mae_train'] = avgs
            performance['mse_train_median'], performance['mae_train_median'] = medians
            performance['mse_train_lower'], performance['mae_train_lower'] = lower_bounds
            performance['mse_train_upper'], performance['mae_train_upper'] = upper_bounds
            performance['mse_train_std'], performance['mae_train_std'] = std
            performance['rmse_train'] = list(np.sqrt(performance['mse_train']))
            performance['rmse_train_median'] = list(np.sqrt(performance['mse_train_median']))
            performance['rmse_train_lower'] = list(np.sqrt(performance['mse_train_lower']))
            performance['rmse_train_upper'] = list(np.sqrt(performance['mse_train_upper']))
            performance['rmse_train_std'] = list(np.sqrt(performance['mse_train_std']))

            # update performances in file
            save_json(performance, performance_file)
        
        
if args.eval_performance_per_sim:
    print('Evaluating model performance per simulation...')
    # read current simulation performances
    performances_file = os.path.join(results_dir, 'performance_per_sim.json')
    performances = load_json(performances_file)
    performances['simulation_name'] = args.simulation_name
    
    performances = performances | {'mse': [], 'rmse': [], 'mae': []}
    
    # update new performance metrics but keep other contents
    for i, sim_dataset in tqdm(enumerate(test_dataset.iterate_simulations(), 1), 
                               total=test_dataset.num_simulations,
                               desc='evaluating simulations', unit='sim'):
        sim_loader = DataLoader(sim_dataset, batch_size=args.batch_size, drop_last=False)
        
        mse, mae = compute_loss(model, sim_loader,
                                [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                sim_dataset.num_samples, args.batch_size, 
                                model_forward_kwargs)
        rmse = np.sqrt(mse)
        
        performances['mse'].append(mse)
        performances['rmse'].append(rmse)
        performances['mae'].append(mae)

    performances['std_mse'] = np.std(performances['mse'])
    performances['std_rmse'] = np.std(performances['rmse'])
    performances['std_mae'] = np.std(performances['mae'])
    
    # update performances in file
    save_json(performances, performances_file)


if args.eval_performance_per_channel:
    if args.loss_on_latent:
        print('Remove `-loss_on_latent` to be able to evaluate the performance per channel.')
    else:
        print('Evaluating model performance per channel...')
        
        # read current performances
        performance_file = os.path.join(results_dir, 'performance_per_channel.json')
        performance = load_json(performance_file)
        performance['simulation_name'] = args.simulation_name
        
        # update new performance metrics but keep other contents
        mse, mae = compute_loss_per_channel(model, test_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()],  
                                            -1, test_dataset.num_samples, args.batch_size,  model_forward_kwargs)
        performance['mse'], performance['mae'] = mse, mae
        performance['rmse'] = list(np.sqrt(mse))
        
        if args.n_train != 0:
            mse_train, mae_train = compute_loss_per_channel(model, train_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()],  
                                                            -1, train_dataset.num_samples, args.batch_size, 
                                                            model_forward_kwargs)
            performance['mse_train'], performance['mae_train'] = mse_train, mae_train
            performance['rmse_train'] = list(np.sqrt(mse_train))
        
        # update performances in file
        save_json(performance, performance_file)


if args.eval_performance_per_height:
    if args.loss_on_latent:
        print('Remove `-loss_on_latent` to be able to evaluate the performance per height.')
    else:
        print('Evaluating model performance per height...')
        # read current performances
        performance_file = os.path.join(results_dir, 'performance_per_height.json')
        performance = load_json(performance_file)
        performance['simulation_name'] = args.simulation_name
        
        # update new performance metrics but keep other contents
        # over all channels:
        mse, mae = compute_loss_per_channel(model, test_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()], -2, 
                                            test_dataset.num_samples, args.batch_size,  model_forward_kwargs)
        performance['mse'], performance['mae'] = mse, mae
        performance['rmse'] = list(np.sqrt(mse))
        
        if args.n_train != 0:
            mse_train, mae_train = compute_loss_per_channel(model, train_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                            -2, train_dataset.num_samples, args.batch_size,  
                                                            model_forward_kwargs)
            performance['mse_train'], performance['mae_train'] = mse_train, mae_train
            performance['rmse_train'] = list(np.sqrt(mse_train))
        
        # per channel:
        mse_per_channel, mae_per_channel = compute_loss_per_channel(model, test_loader, 
                                                                    [torch.nn.MSELoss(), torch.nn.L1Loss()],  
                                                                    [-1, -2], test_dataset.num_samples, 
                                                                    args.batch_size,  model_forward_kwargs)
        performance['mse_per_channel'], performance['mae_per_channel'] = mse_per_channel, mae_per_channel
        performance['rmse_per_channel'] = np.sqrt(mse_per_channel).tolist()
        
        if args.n_train != 0:
            mse_train_per_channel, mae_train_per_channel = compute_loss_per_channel(
                model, train_loader, [torch.nn.MSELoss(), torch.nn.L1Loss()], [-1, -2], 
                train_dataset.num_samples, args.batch_size,  model_forward_kwargs)
            performance['mse_train_per_channel'] = mse_train_per_channel
            performance['mae_train_per_channel'] = mae_train_per_channel
            performance['rmse_train_per_channel'] = np.sqrt(mse_train_per_channel).tolist()
        
        # update performances in file
        save_json(performance, performance_file)


if args.check_equivariance:
    # TODO save to file
    if isinstance(model, escnn.nn.EquivariantModule):
        print('Checking Equivariance...')
        model.check_equivariance(gpu_device=DEVICE, atol=1e-3) 
    else:
        # TODO implement computation of equivariance error for other models
        print('Equivariance can currently only be checked for steerable models.')
        
        
if args.animate2d:
    anim_dir = os.path.join(results_dir, 'animations')
    horizontal_size = int(args.simulation_name.split('_')[0][1:])
    height = int(args.simulation_name.split('_')[2][1:])
    
    print('Animating 2D...')
    with tqdm(total=12, desc='animating', unit='animation') as pbar:
        for feature in ['t', 'u', 'v', 'w']:
            for axis, dim in enumerate(['width', 'depth', 'height']):
                slice = height//2 if axis == 2 else horizontal_size//2
                
                if is_forecast_model:
                    assert not args.loss_on_latent, 'animation is performed on original representation, \
                        set loss_on_latent to False'
                    visualization.rb_forecaster_animation(slice=slice, fps=24, num_forecasts=args.animation_samples, 
                                        feature=feature, axis=axis, anim_dir=os.path.join(anim_dir, feature), 
                                        anim_name=f'{dim}.mp4', model=model, sim_file=anim_sim_file, device=DEVICE,
                                        warmup_seq_length=args.autoregressive_warmup_seq_length)
                else:
                    visualization.rb_autoencoder_animation(slice=slice, fps=24, frames=args.animation_samples,
                                        feature=feature, axis=axis, anim_dir=os.path.join(anim_dir, feature), 
                                        anim_name=f'{dim}.mp4', model=model, sim_file=anim_sim_file, device=DEVICE)
                pbar.update(1)
                
                
if args.animate3d:
    anim_dir = os.path.join(results_dir, 'animations')
    
    print('Animating 3D...')
    with tqdm(total=4, desc='animating', unit='animation') as pbar:
        for feature in ['t', 'u', 'v', 'w']:
            if is_forecast_model:
                assert not args.loss_on_latent, 'animation is performed on original representation, \
                    set loss_on_latent to False'
                visualization.rb_forecaster_animation_3d(fps=24, num_forecasts=args.animation_samples, feature=feature, 
                                           anim_dir=os.path.join(anim_dir, feature), anim_name=f'3d.mp4', 
                                           model=model, sim_file=anim_sim_file, device=DEVICE, contour_levels=50,
                                           warmup_seq_length=args.autoregressive_warmup_seq_length)
            else:
                visualization.rb_autoencoder_animation_3d(fps=24, frames=args.animation_samples, feature=feature, 
                                            anim_dir=os.path.join(anim_dir, feature), anim_name=f'3d.mp4', 
                                            model=model, sim_file=anim_sim_file, device=DEVICE, contour_levels=50)
            pbar.update(1)


if args.compute_latent_sensitivity:
    print('Computing latent sensitivity...')
    if is_forecast_model:
        print('Latent sensitivtiy can only be computed for autoencoders')
    else:
        sensitivity_dataset = dataset.RBDataset(sim_file, 'test', device=DEVICE, shuffle=True)
        if (args.latent_sensitivity_samples > 0):
            samples = min(sensitivity_dataset.num_samples, args.latent_sensitivity_samples) 
        else:
            samples = sensitivity_dataset.num_samples
        
        avg_sensitivity, avg_abs_sensitivity = compute_latent_sensitivity(
            model, sensitivity_dataset, samples=samples, save_dir=results_dir,
            filename='latent_sensitivity', parallel_channels=args.parallel_channels_latent_sensitivity)