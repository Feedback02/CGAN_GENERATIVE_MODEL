# Manga Colorization using CGAN with U-Net Generator and PatchGAN Discriminator

## Overview

This project focuses on the colorization of manga images using Conditional Generative Adversarial Networks (CGAN). We utilize a U-Net architecture for the generator and a PatchGAN architecture for the discriminator. Instead of the conventional RGB color space, this project employs the L*a*b* color space to potentially enhance the colorization quality.

## Introduction

Manga, a style of Japanese comic books and graphic novels, is typically black and white. Colorizing manga can make it more visually appealing and accessible to a wider audience. This project leverages a CGAN approach to automatically colorize manga images, enhancing their visual quality with minimal manual effort.

## Dataset

The dataset used for this project can be downloaded from the following link:

[Download Dataset](https://drive.google.com/file/d/1aM8RTM3rgVFkUUp-ppsnhfWD7AadR2Sp/view?usp=sharing)

The dataset contains grayscale manga images along with their corresponding colorized versions.

## Architecture

### Generator

The generator is based on the U-Net architecture, which is known for its effectiveness in image-to-image translation tasks. U-Net consists of an encoder-decoder structure with skip connections, allowing it to capture both global context and fine details.

### Discriminator

The discriminator is a PatchGAN, which discriminates on a patch level rather than the entire image. This helps in focusing on local image features, improving the overall realism of the generated images.

### Color Space

Instead of using the traditional RGB color space, this project uses the L*a*b* color space. The L*a*b* color space separates the lightness (L*) from the color information (a* and b*), which can be beneficial for colorization tasks.

## Installation

To run this project, you'll need Python and several Python libraries. The following instructions will guide you through the installation process:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/manga-colorization-cgan.git
    cd manga-colorization-cgan
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to reach out if you have any questions or need further assistance. Enjoy colorizing your manga!
