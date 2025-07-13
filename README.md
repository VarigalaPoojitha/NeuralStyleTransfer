# Neural Style Transfer

A Python implementation of Neural Style Transfer using TensorFlow and VGG19. This project applies the artistic style of one image to the content of another image using deep learning techniques.

## Features

- Neural Style Transfer using VGG19 architecture
- Customizable style and content layers
- Automatic image preprocessing and resizing
- Real-time style transfer with progress tracking

## Files

- `Neural_Style_Transfer.py`: Main script for neural style transfer
- `content.jpg`: Content image used for style transfer
- `style.jpg`: Style image used for style transfer
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore rules

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VarigalaPoojitha/NeuralStyleTransfer.git
cd NeuralStyleTransfer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your content image as `content.jpg` and style image as `style.jpg` in the project directory
2. Run the script:
```bash
python Neural_Style_Transfer.py
```

The script will:
- Load and preprocess both images
- Apply neural style transfer using VGG19
- Display the stylized result
- Print the processing time

## Parameters

You can modify the following parameters in the script:
- `epochs`: Number of training iterations (default: 10)
- `max_dim`: Maximum dimension for image resizing (default: 512)
- Learning rate and other optimization parameters

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Pillow

## License

This project is open source and available under the MIT License. 