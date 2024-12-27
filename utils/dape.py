import torch
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        A simple dataset to load images from a folder.

        :param image_folder: Path to the folder containing images.
        :param transform: Transformations to apply to the images.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                            if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class ImageToTextGenerator:
    def __init__(self, model_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/ram_swin_large_14m.pth', condition_path='/home/dongnan/SLF/NVC/DiffEIC-main/weight/models/DAPE.pth'
                 ,device='cuda'):
        """
        Initializes the ImageToTextGenerator with the RAM model.

        :param model_path: Path to the pretrained RAM model weights.
        :param condition_path: Path to the pretrained condition weights.
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.weight_dtype = torch.float16

        # Load the RAM model
        self.model = ram(pretrained=model_path,
                         pretrained_condition=condition_path,
                         image_size=384,
                         vit='swin_l')
        self.model.eval()
        self.model.to(self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def generate_text_from_tensor(self, input_tensor, user_prompt='', positive_prompt=''):
        """
        Generates text descriptions from an input tensor.

        :param input_tensor: Input tensor of shape (b, 3, h, w)
        :param user_prompt: Additional user-provided prompt.
        :param positive_prompt: Positive prompt to enhance descriptions.
        :return: List of generated text descriptions.
        """
        # Move tensor to the specified device and dtype
        lq = input_tensor.to(self.device).type(self.weight_dtype)

        # Resize images to 384x384 if necessary
        b, c, h, w = lq.shape
        if h != 384 or w != 384:
            lq = torch.nn.functional.interpolate(lq, size=(384, 384), mode='bilinear', align_corners=False)
        print("lq.shape:",lq.shape)
        # Generate descriptions using the RAM model
        res = inference(lq, self.model)
        # Construct the final text prompts
        generated_texts = []
        for i in range(len(res)):
            validation_prompt = f"{res[i]}, {positive_prompt},"
            if user_prompt != '':
                validation_prompt = f"{user_prompt}, {validation_prompt}"
            generated_texts.append(validation_prompt)
        txt_list = generated_texts[0]
        txt_list = txt_list.rstrip(', ')
        txt_cleaned = txt_list.strip("[]").replace("'", "")
        return generated_texts
"""
# 写一个简单数据加载，给出使用示例，在源代码上加
if __name__ == "__main__":
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    image_folder = '/home/dongnan/SLF/data/pic_coms/valid_o'  # Replace with your image folder path
    dataset = ImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Initialize the generator
    generator = ImageToTextGenerator(model_path='weight/models/ram_swin_large_14m.pth',condition_path='weight/models/DAPE.pth')

    # Generate descriptions for images in the dataloader
    for batch in dataloader:
        generated_texts = generator.generate_text_from_tensor(batch)
        print(generated_texts[0])
        exit()
"""