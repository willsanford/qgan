# Append parent directory for imports
import os, sys, subprocess
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# General Imports
import lib
from trainer import Trainer
from logger import Logger
from datetime import datetime

# Check for proper meta file
if len(sys.argv) != 2:
    print('Incorrect arguemnts. Please use "training.py meta.txt" with your desired meta file.') 
    exit()
meta = lib.load_meta(sys.argv[1])

# Ensure that the correct folders exist
if not os.path.isdir('results'):
    subprocess.run(['mkdir', 'results'])
if not os.path.isdir(os.path.join('results', meta['run_name'])):
    subprocess.run(['mkdir', os.path.join('results', meta['run_name'])])

# Initialize the logger
l = Logger(file = os.path.join('results', meta['run_name'], meta['log_file_name'] + datetime.now().strftime("%m-%d_%H:%M:%S") + '.txt'), 
           include = meta['include'].split('/'),
           stdout = bool(int(meta['log_out'])))
l.open()
l.log('LOG', f'Meta loaded using file: {sys.argv[1]}')


# Get Cuda Device
device = lib.get_cuda()
l.log('LOG', f'CUDA {"Loaded" if device != "cpu" else "not loaded"}. Using device {device}')

# Set up the generic trainer
trainer = Trainer(
    steps = int(meta['steps']),
    model = lib.get_model(meta['model'], l, device),
    logger = l,
    epochs = int(meta['epochs']),
    save_path = meta['save_path'],
    save_name = meta['save_name'],
    load_name = meta['load_name'],
    cuda = bool(int(meta['cuda']))
)

# Apply all relavent modes
modes = meta['modes'].split('/')

if 'train_model' in modes:
    trainer.train()
if 'visualize' in modes:
    lib.visualize_losses(os.path.join('results', meta['run_name']), trainer.get_losses())
