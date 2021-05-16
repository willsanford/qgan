# Append parent directory for imports
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# General Imports
from common import load_meta, log
from trainer import Trainer
import torch.optim as optim

# Check for proper meta file
if len(sys.argv) != 2:
    log('ERR', 'Incorrect arguemnts. Please use "training.py meta.txt" with your desired meta file.') 
    exit()
meta = load_meta(sys.argv[1])
log('LOG', f'Meta loaded using file: {sys.argv[1]}')


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
    real_fake_threshold = float(meta['rf_threshold']),
    epochs = int(meta['epochs']),
    steps_per_checkpoint = int(meta['steps_per_checkpoint']),
    save_path = meta['save_path'],
    save_name = meta['save_name'],
    load_name = meta['load_name'],
    cuda = bool(int(meta['cuda']))
)

# Start training sequence
trainer.train()
