import sys
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class Tester:

    def __init__(self, model, dataloader, len_dataset, img_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.len_dataset = len_dataset
        self.img_size = img_size

    def __call__(self):
        
        self.model.eval()

        tot_correct_predictions = 0

        for images, targets in tqdm(self.dataloader, position=0, leave=True, file=sys.stdout):

            images = images.to(self.device)

            targets = targets.long()
            targets = torch.squeeze(targets, 1)
            targets = targets.to(self.device)

            output_probs = self.model(images)
            predictions = torch.argmax(output_probs, dim=1)
            # predictions.apply_(lambda x: x if x!=21 else 255)

            correct_predictions = torch.sum(predictions == targets).item()
            tot_correct_predictions += correct_predictions

        accuracy = tot_correct_predictions/(self.len_dataset*(self.img_size**2))*100

        return accuracy

class TestOne:
    
    def __init__(self, model, dataset, dataset_nonorm):
        
        self.model = model
        self.dataset = dataset
        self.dataset_nonorm = dataset_nonorm
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        palette = torch.tensor([2**25-1, 2**15-1, 2**21-1])
        self.colors = torch.as_tensor([i for i in range(21)])[:, None]*palette
        self.colors = (self.colors%255).numpy().astype("uint8")

    def __call__(self, img_id):

        self.model.eval()

        true_img, _ = self.dataset_nonorm[img_id]
        img, target = self.dataset[img_id]

        img = img.to(self.device)
        target = target.to(self.device)

        output_probs = self.model(img.unsqueeze(0))

        prediction = torch.argmax(output_probs, dim=1)
        prediction = prediction.squeeze(0)

        fig, ax = plt.subplots(1,3, figsize=(14,10))

        ax[0].imshow(transforms.ToPILImage()(true_img))

        img_to_show = target.byte().cpu().numpy()
        r = Image.fromarray(img_to_show)
        r = r.resize(target.size())
        r.putpalette(self.colors)

        ax[1].imshow(r)

        img_to_show = prediction.byte().cpu().numpy()
        r = Image.fromarray(img_to_show)
        r = r.resize(prediction.size())
        r.putpalette(self.colors)

        ax[2].imshow(r)

        plt.show()
