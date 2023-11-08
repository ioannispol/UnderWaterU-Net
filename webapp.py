import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image

from underwater_unet.model import UNet

available_models = ["UW-Unet", "model1", "model2"]


# Define a function to load the model
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


# Define a function that will take an image input and predict the output using the trained model
def predict(image, model_name):
    if image is None:
        raise ValueError("No image provided.")

    if model_name == "UW-Unet":
        # Load the UW-Unet model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model('experiment/exp_22g93hvt/model_epoch_8.pth')
    else:
        raise ValueError("Model not found.")

    # Convert the PIL Image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations that were used in the training phase
    ])

    image = transform(image).unsqueeze(0).to(device)

    # Predict the mask
    with torch.no_grad():
        output = model(image)

    # Assuming a single-channel output, apply sigmoid and convert to an image
    # If you have multi-class output, you'll need to adjust this
    predicted_mask = torch.sigmoid(output[0, 0]).cpu().numpy()
    predicted_mask = (predicted_mask * 255).astype('uint8')
    mask_image = Image.fromarray(predicted_mask)

    return mask_image


# Load your trained model (adjust the path to your .pth file)
model_path = 'experiment/exp_22g93hvt/model_epoch_8.pth'
model = load_model(model_path)

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(),  # Adjust the shape to match your model's input shape
        gr.Dropdown(choices=available_models, value="UW-Unet")
    ],
    outputs='image',
    examples=[
        ['data/images/076193.jpg', 'UW-Unet'],
        ['data/images/076350.jpg', 'UW-Unet']
    ],
    title="Image Segmentation with UNet",
    description="Upload an image and select a model to predict the segmentation mask."
)
# ['data/images/076193.jpg', 'data/images/076350.jpg']

# Launch the interface
iface.launch()
