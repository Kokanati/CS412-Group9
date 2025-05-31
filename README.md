# Phishing Website Detection Using MCMC-GAN and Stacking Ensemble

**Developed by: CS412 Group 9**  
**Academic Year: 2025**

This repository is the working directory for Group 9. 

---
## For TESTING PURPOSES
To download the working directory along with the Proposed_Solution
```bash
git clone https://github.com/Kokanati/CS412-Group9.git
cd CS412-Group9
```

## Dataset Directory
Contains the raw_dataset.csv file and use the script to modify the dataset to a more realistic imbalance. It will generate 3 versions of imbalance, 80/20, 90/10 and 95/5.

To run the script 
python undersample.py

it will generate 3 files then run the feature_selection.py. This will apply dimension modification and generate 3 different versions of 30 features, 20 and 10.
You should now have 9 sets of dataset to use for testing.
---

## Environment Setup

### Step 1: Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Running MCMC-GAN test
Make sure all the 9 newly generated datasets are copied to the directory 'dataset'
run the oneClick-BatchRunner.py

```bash
python oneClick-BatchRunner.py
```
This will take all the datasets in the dataset directory and feed it into the MCMC-GAN.py script.

For output
- All logs will be saved in the logs directory
- outputs directory will have all the evaluation results from the run being saved in each directory named after the dataset

### SMOTIFIED-GAN
This script was used to run using google colab.
Upload all the datasets and run the code and a prompt will ask you for the dataset


## SMOTE
This script was used to run using google colab.
Upload all the datasets and configure path to dataset to be run one by one

## Proposed_Solution
This directory contains the Groups solution for their problem statement saved in the phishing-detector directory
