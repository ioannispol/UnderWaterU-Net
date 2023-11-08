import gradio as gr
import torch

from underwater_unet.model import UNet
from prediction import predict_image, mask_to_image

models = {
    "UW-Unet": "experiment/exp_22g93hvt/model_epoch_8.pth",
    "UW-Unet1": "experiment/exp_22g93hvt/model_epoch_4.pth",
    "UW-Unet2": "experiment/exp_22g93hvt/model_epoch_2.pth"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define a function to load the model
def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model


loaded_models = {name: load_model(path) for name, path in models.items()}


# Define a function that will take an image input and predict the output using the trained model
def predict(image, model_name):
    model = loaded_models[model_name]  # Get the model based on the model name
    output = predict_image(model, image, device)
    mask_image = mask_to_image(output)

    return mask_image


# Load trained model (adjust the path to your .pth file)
model_path = 'experiment/exp_22g93hvt/model_epoch_8.pth'
model = load_model(model_path)

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(),  # Adjust the shape to match your model's input shape
        gr.Dropdown(choices=models, value="UW-Unet")
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
iface.launch(share=True)
