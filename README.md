# AntiBinder
AntiBinder: a sequence-structure hybrid model based on bidirectional cross-attention mechanism for predicting antibody-antigen binding relationships.

![framework](./figures/model_all.png)

## Introduction
This project is used for predicting antigen-antibody affinity for protein types. The model can be trained and used based solely on sequence data. You can also stack the modules within, making the model's parameters significantly larger, and train it to achieve a plug-and-play effect.

## Dependencies
python 3.11

## Installation Guide
Detailed instructions on how to install and set up the project:

### Clone the repository
git clone https://github.com/brisingr10/AntiBinder2.git

### Install dependencies
pip install -r requirements.txt

### Usage Instructions
#### Training
Prepare labeled sequence data for antigens and antibody heavy chains. Name the columns according to the names specified in the `antigen_antibody_emb.py`. If the heavy chain sequences of the antibodies have not been split, first use `heavy_chain_split.py` to split the sequences. Then, use the command: `python main_trainer.py` to start the model training.

```python
# Example code for starting training
# Start training
python main_trainer.py
```
#### Testing
Prepare labeled sequence data for antigens and antibodies. Name the columns according to the names specified in the `antigen_antibody_emb.py`. Then, use the command: `python main_test.py` to start the test.
```python
# Example code for starting testing

# Start training
python main_test.py --input_path "data/test.csv" --checkpoint_path "models/my_model.pth"
```
