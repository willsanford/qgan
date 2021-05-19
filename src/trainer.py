#
# Trainer
#
import torch
import os, time, datetime
from logger import Logger

class Trainer():
    def __init__(self,
                 steps: int,
                 model,
				 logger: Logger,
                 epochs: int,
                 save_path: str,
                 save_name: str,
                 load_name: str = 'Null',
                 cuda: bool = True):

        # Training parameters
        self.steps = steps
        self.epochs = epochs

        # Model Params
        self.model = model

        # Logger
        self.l = logger

        # Check accelerator compatability and send the network to the compatible device
        self.device = self._check_cuda(cuda)
        self.l.log('LOG', f'Trainer loaded using device: {torch.cuda.get_device_name(device=self.device)}')

        # Model check point parameters
        self.save_name = save_name
        self.save_path = save_path
        self.load_name = load_name

		# Save model losses
        self.g_loss = []
        self.d_loss = []

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

    def get_losses(self):
        return self.g_loss, self.d_loss
        
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
        
        self.l.log('LOG', 'Training started')
        while epoch < self.epochs:
            epoch += 1
            step = 0
            self.l.log('LOG', f'Starting epoch: {epoch}')
            while step < self.steps:
                step +=1
                d_loss, g_loss = self.model.step()
                self.g_loss.append(g_loss), self.d_loss.append(d_loss)
                self.l.log('TRN', 'Epoch: %3d Step: %4d  Gen Loss: %.4f  Disc Loss: %.4f' %(epoch, step, g_loss, d_loss))

			# Save the model dictionary in the path {save_path}/{save_name}_{g|d}{epoch}
            g = self.model.save_checkpoint('gen', os.path.join(self.save_path, self.save_name + 'g' + str(epoch)))
            d = self.model.save_checkpoint('disc', os.path.join(self.save_path, self.save_name + 'd' + str(epoch)))
            if g and d:
                self.l.log('TRN', f'Weights logged for epoch {epoch}')
            else:
                self.l.log('ERR', 'Couldn\'t save checkpoint')
			
              
        self.l.log('LOG', f'Training has finished. Completed in {datetime.timedelta(seconds=(time.time() - start_time))}.')
