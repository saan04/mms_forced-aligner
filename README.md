# MMS Forced Aligner for Kaggle

This repository contains a Kaggle-optimized implementation of the Multilingual Multi-Speaker (MMS) Forced Aligner, originally developed by AI4Bharat. This adaptation allows for efficient audio-text alignment processing within Kaggle's computational environment. The above code runs well even in a local computer setup (no GPU) but the output would only be genarated after 5-10 minutes of processing.

## Overview

The script in this repository:
1. Sets up the necessary environment by cloning uroman and installing required dependencies. (sym link creation error noticed)
2. Implements the MMS forced aligner for audio-text alignment.
3. Processes audio files in batches to manage Kaggle's output capacity.
4. Generates and saves alignment results in JSON format.

## Key Features

- **Kaggle Optimization**: Tailored to run efficiently in Kaggle notebooks.
- **Batch Processing**: Allows for processing subsets of audio files to manage output capacity.
- **Automatic Setup**: Clones required repositories and installs dependencies.
- **Audio File Format**: Conversion of audio file to the standard wav format with 16kHz and 256kbps.

## Prerequisites

- Kaggle notebook environment
- Input audio files in WAV format
- Corresponding text transcripts

## Usage

1. Upload the script to a Kaggle notebook.
2. Ensure your audio files and text transcripts are available in the specified Kaggle directories.
3. Run the notebook cells sequentially.

## File Structure
We are writing them into the kaggle notebook for easier flow of code execution. 
- `align.py`: Main script for the forced alignment process.
- `punctuations.lst`: List of punctuations used in the alignment process.

## Customization

- Adjust input and output directory paths as needed for your Kaggle setup.
- Creation of a "temp" directory to utilise more storage and not max out as suggested in "https://www.kaggle.com/discussions/product-feedback/372506"
- Make sure to specify the language code while calling the command "mni" in this use case.

## Acknowledgments

This project is an adaptation of the MMS Forced Aligner by AI4Bharat. Visit their [GitHub repository](https://github.com/AI4Bharat/ai4b-fairseq/blob/main/examples/mms/data_prep) for more information on the original implementation.
