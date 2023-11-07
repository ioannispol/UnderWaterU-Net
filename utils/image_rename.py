import os
import argparse
import re


def rename_files(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compile a regular expression pattern to match the number in the file names
    pattern = re.compile(r'adjusted_(\d+)_segmentation\.png')

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        match = pattern.match(file_name)
        if match:
            new_name = f'mask-{match.group(1)}.png'
            original_path = os.path.join(input_folder, file_name)
            new_path = os.path.join(output_folder, new_name)
            os.rename(original_path, new_path)
            print(f'Renamed {file_name} to {new_name}')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Rename files according to a specified pattern.')
    parser.add_argument('input_folder', help='The folder containing the files to be renamed.')
    parser.add_argument('output_folder', help='The folder where the renamed files will be placed.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Run the rename function
    rename_files(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
