#
# Trainer
#
import torch

from typing import Dict
from tqdm import tqdm
import os, time, datetime
from common import log
import random

class Trainer():
    def __init__(self,
                 steps: int,
                 model,
                 real_fake_threshold: float,
                 epochs: int,
                 steps_per_checkpoint: int,
                 save_path: str,
                 save_name: str,
                 load_name: str = 'Null',
                 cuda: bool = True):

        # Training parameters
        self.steps = steps
        self.epochs = epochs
        self.threshold = real_fake_threshold

        # Model Params
        self.model = model

        # Check accelerator compatability and send the network to the compatible device
        self.device = self._check_cuda(cuda)
        log('LOG', f'Trainer loaded using device: {torch.cuda.get_device_name(device=self.device)}')

        # Model check point parameters
        self.spc = steps_per_checkpoint
        self.save_name = save_name
        self.save_path = save_path
        self.load_name = load_name

    def _check_cuda(self, cuda: bool) -> str:
        '''
          Checks the availablity of a CUDA enabled accelerator. If one isn't available, raise an error. We dont currently allow for distributed training. This is coming in the future

          Args:
            cuda: boolean describing whether the user has an available accelerator

          Returns:
            torch_device: the string representing the device to be used by torch
        '''
        if not cuda:
            return 'cpu'
        elif not torch.cuda.is_available():
            raise Exception('You claim to have a CUDA enabled device, but pytorch cannot find it')
        else:
            return 'cuda:0'

    # TODO: ADD LOGGING HERE
    def _load_ckpt(self):
      '''
      If a load path is given, load the most recent iteration of the model
      '''
      if self.load_name != 'Null':
        ckpt = os.listdir(self.save_path)
        latest_ckpt = sorted(ckpt, key= lambda fname: int(fname.split('_')[1]))[0]
        self.net.load_state_dict(torch.load(os.path.join(self.load_path, latest_ckpt)))    
      else:
        pass
    def train(self):
        '''
        This function handles all of training logic.

        Args: 
        '''
        self._load_ckpt()
        
        start_time = time.time()        
        epoch = 0
        
        log('LOG', 'Training started')
        while epoch < self.epochs:
            epoch += 1
            step = 0
            log('LOG', f'Starting epoch: {epoch}')
            while step < self.steps:
                step +=1
                d_loss, g_loss = self.model.step()
                log('TRN', 'Epoch: %3d Step: %4d  Gen Loss: %.4f  Disc Loss: %.4f' %(epoch, step, g_loss, d_loss))

                        # Save the model dictionary in the path {save_path}/{save_name}_{step number}
                        # TODO: Log this properly 
                        # if step % self.spc == 0:
                        #     torch.save(self.net, os.path.join(self.save_path, self.save_name + '_' +  str(step)))
              
        log('LOG', f'Training has finished. Completed in {datetime.timedelta(seconds=(time.time() - start_time))}.')
