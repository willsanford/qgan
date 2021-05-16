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
                 steps_per_checkpoint: int,
                 save_path: str,
                 save_name: str,
                 load_name: str = 'Null',
                 cuda: bool = True):

        # Training parameters
        self.steps = steps
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

    # TODO: This funcitonality was moved into the models. This can likely be removed
    def _train_step(self, inputs: Dict[str, torch.tensor]) -> torch.tensor:
        '''
          Runs a single training step a single batch of inputs. 

          Args:
            inputs: This is a dictionary of the inputs of the given DAG. This should match the naming scheme of the defined DAG

          Returns:
            loss: returns outputs of the network run
        '''
        return self.net(inputs)
    
    def _load_ckpt(self):
      '''
      If a load path is given, load the most recent iteration of the model
      '''
      if self.load_name != 'Null':
        ckpt = os.listdir(self.save_path)
        latest_ckpt = sorted(ckpt, key= lambda fname: int(fname.split('_')[1]))[0]
        self.net.load_state_dict(torch.load(os.path.join(self.load_path, latest_ckpt)))    

    def train(self):
        '''
        This function handles all of training logic.

        Args: 
        '''
        self._load_ckpt()
        
        start_time = time.time()        
        epoch = 0
        step = 0

        log('LOG', 'Training started')

        while step < self.steps:
            epoch += 1
            log('LOG', f'Starting Epoch {epoch}')
            step +=1
            print(self.model)
            d_loss, g_loss = self.model.step()
            log('LOG', f'STEP: {step}')


            # with tqdm(enumerate(dataloader)) as t:
            #     for batch_num, batch_tuple in t:
            #         batch, labels = batch_tuple
            #         # Cuda enable any input tensors if using a cuda device and then send all the data to the proper device
            #         if self.device != 'cpu':
            #             [batch[k].cuda() for k in batch.keys()]
            #         [batch[k].to(self.device) for k in batch.keys()]

                    # Check if we will be using real or generated data

                    # TODO: Moved the logic that was here to inside the model class. rewrite what was here to reflect 

                    #  t.set_description('Step: %6d Epoch: %4d Batch: %4d Loss: %.3f' %(step + 1, epoch, batch_num, loss))

                    

                    # Save the model dictionary in the path {save_path}/{save_name}_{step number}
                    # TODO: Log this properly 
                    # if step % self.spc == 0:
                    #     torch.save(self.net, os.path.join(self.save_path, self.save_name + '_' +  str(step)))
          
        log('LOG', f'Training has finished. Completed in {datetime.timedelta(seconds=(time.time() - start_time))}.')
