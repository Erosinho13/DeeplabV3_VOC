import sys
import torch
from tqdm import tqdm

class Trainer:

    def __init__(self, model, dataloader, optimizer_fn, optim_scheduler, loss_fn, len_dataset, img_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer_fn = optimizer_fn
        self.optim_scheduler = optim_scheduler
        self.loss_fn = loss_fn.to(self.device)
        self.len_dataset = len_dataset
        self.img_size = img_size
        self.iter_per_epoch = len(dataloader)

    def __call__(self):

        self.model.train()

        epoch_loss = 0
        tot_correct_predictions = 0

        for images, targets in tqdm(self.dataloader, position=0, leave=True, file=sys.stdout):
            
            images = images.to(self.device)

            targets = targets.long()
            targets = torch.squeeze(targets, 1)
            targets = targets.to(self.device)

            self.optimizer_fn.zero_grad()
            output_probs = self.model(images)

            loss = self.loss_fn(output_probs, targets)
            loss.backward()
            self.optimizer_fn.step()
            
            predictions = torch.argmax(output_probs, dim=1)
            # predictions.apply_(lambda x: x if x!=21 else 255) #works only with cpu tensors

            correct_predictions = torch.sum(predictions == targets).item()
            tot_correct_predictions += correct_predictions

            epoch_loss += loss.item()
            
        self.optim_scheduler.step()
        
        avg_loss = epoch_loss/self.iter_per_epoch
        accuracy = tot_correct_predictions/(self.len_dataset*(self.img_size**2))*100

        return avg_loss, accuracy