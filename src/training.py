# Append parent directory for imports
import os, sys, subprocess
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# General Imports
from common import load_meta
from trainer import Trainer
from logger import Logger
from datetime import datetime

# Check for proper meta file
if len(sys.argv) != 2:
    print('Incorrect arguemnts. Please use "training.py meta.txt" with your desired meta file.') 
    exit()
meta = load_meta(sys.argv[1])
print(meta)
# Ensure that the correct folders exist
if not os.path.isdir('results'):
    subprocess.run(['mkdir', 'results'])
if not os.path.isdir(os.path.join('results', meta['run_name'])):
    subprocess.run(['mkdir', os.path.join('results', meta['run_name'])])

# Initialize the logger
l = Logger(file = os.path.join('results', meta['run_name'], meta['log_file_name'] + datetime.now().strftime("%m-%d_%H:%M:%S") + '.txt'), 
           include=meta['include'].split('/'),
           stdout=bool(int(meta['log_out'])))
l.open()
l.log('LOG', f'Meta loaded using file: {sys.argv[1]}')

# Model imports and Dictionary, Allows models to be loaded from meta file
from models.basic_q import InitialQuantumModel
from models.test_models import TestDisc, TestGen
model_dic = {
    'basic_q': InitialQuantumModel()
}

# Set up the generic trainer
trainer = Trainer(
    steps = int(meta['steps']),
    model = model_dic[meta['model']],
    logger = l,
    real_fake_threshold = float(meta['rf_threshold']),
    epochs = int(meta['epochs']),
    save_path = meta['save_path'],
    save_name = meta['save_name'],
    load_name = meta['load_name'],
    cuda = bool(int(meta['cuda']))
)

# Start training sequence
trainer.train()
