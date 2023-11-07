import os
import argparse
from PIL import Image


def convert_tif_to_png(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.tif'):
            # Construct full file path
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name.lower().replace('.tif', '.png'))

            # Open the tif file
            with Image.open(input_file_path) as img:
                # Save as PNG
                img.save(output_file_path, 'PNG')
            print(f'Converted {file_name} to PNG format.')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert TIF images to PNG.')
    parser.add_argument('input_folder', help='The folder containing TIF images.')
    parser.add_argument('output_folder', help='The folder where PNG images will be saved.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Use the function to convert all .tif images
    convert_tif_to_png(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
