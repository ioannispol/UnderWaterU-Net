# UnderWaterU-Net 🌊

![UnderWaterU-Net Logo](path_to_my_logo.png) 
<!-- Replace with a logo in the path here. -->

Welcome to UnderWaterU-Net, a deep learning repository specially optimized for underwater image segmentation. With challenges like inconsistent lighting, suspended particles, and the dynamic nature of the underwater environment, traditional image segmentation models often fall short. Enter UnderWaterU-Net: a tailored solution designed with the depths in mind.

## 🌟 Features

- **Tailored U-Net Architecture**: Customized to perform optimally on underwater images.
- **Expandable with Submodules**: Modular design allows for easy expansion and incorporation of additional functionalities.
- **Streamlined Workflow**: From raw underwater images to precise segmentations, UnderWaterU-Net makes the process seamless.


## 🚀 Getting Started

### Prerequisites

- List any prerequisites or dependencies here.

### Installation

1. **Direct Installation**:
   ```bash
   git clone git@github.com:ioannispol/UnderWaterU-Net.git
   ```

2. **Advanced Setup (With Submodules)**:
   ```bash
   git clone --recurse-submodules git@github.com:ioannispol/UnderWaterU-Net.git
   ```

## 📖 Documentation

Detailed documentation can be found [here](link_to_your_documentation). 
<!-- Replace with a link to your documentation if you have it. -->

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](link_to_contributing_guide) for details. 
<!-- Replace with a link to your contributing guide if you have it. -->

## 📜 License

This project is licensed under the XYZ License - see the [LICENSE.md](link_to_license) for details. 
<!-- Replace with a link to your license file and mention the type of license you're using. -->

## 📬 Contact

For any queries, feel free to reach out to [ioannispol](mailto:your_email@example.com). 
<!-- Replace with your email or contact details. -->

## Attention Mechanisms in U-Net

The U-Net architecture has been extended to include attention gates, which allow the model to focus on specific regions of the input, enhancing its capability to segment relevant regions more accurately.

### AttentionGate Module

The AttentionGate module takes two inputs, \( g \) and \( x \), and computes the attention coefficients. These coefficients are used to weight the features in \( x \) to produce the attended features. The process can be summarized as follows:

1. Two 1x1 convolutions transform \( g \) and \( x \) into a compatible space.
2. A non-linearity (ReLU) is applied after summing the transformed versions of \( g \) and \( x \).
3. Another 1x1 convolution followed by a sigmoid activation produces the attention coefficients in the range [0, 1].
4. The original \( x \) is multiplied by the attention coefficients to obtain the attended features.

This mechanism is particularly useful in tasks like image segmentation, enabling the network to emphasize more informative regions during training and prediction.

### Reference

The attention mechanism is inspired by the following paper:
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Glocker, B. (2018). Attention U-Net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.


