import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from underwater_unet.model import UNet


def predict_image(model, image, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations that were used in the training phase
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    return output


def mask_to_image(mask):
    # Convert the predicted mask to an image
    predicted_mask = torch.sigmoid(mask[0, 0]).cpu().numpy()
    predicted_mask = (predicted_mask * 255).astype('uint8')
    mask_image = Image.fromarray(predicted_mask)
    return mask_image


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    return parser.parse_args()


def get_output_filenames(input_list, output_list):
    if output_list is None:
        return [f'{os.path.splitext(f)[0]}_OUT.jpg' for f in input_list]
    return output_list


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    output_files = get_output_filenames(args.input, args.output)

    for i, input_path in enumerate(args.input):
        image = Image.open(input_path).convert('RGB')
        output = predict_image(model, image, device)
        mask_image = mask_to_image(output)
        mask_image.save(output_files[i])
