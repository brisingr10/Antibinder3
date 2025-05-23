"""
This script is designed to analyze and visualize amino acid sequence data.
"""
import os
import random
import logomaker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


def generate_seq_records(dataframe):
    record_list = []
    for idx, data_row in dataframe.iterrows():
        seq_entry = SeqRecord(Seq(data_row['vh']), id=str(idx))
        record_list.append(seq_entry)
    return record_list

def perform_alignment(seq_records):
    with open("temp.fasta", "w") as tmp_file:
        for seq_entry in seq_records:
            tmp_file.write(f">{seq_entry.id}\n")
            tmp_file.write(str(seq_entry.seq) + "\n")
    
    alignment_command = "muscle5.1.linux_intel64 -align temp.fasta -output aln.fasta"
    os.system(alignment_command)
    alignment_result = AlignIO.read("aln.fasta", "fasta")
    return alignment_result

def extract_aligned_sequences(aligned_data):
    aligned_seqs = []
    for alignment_record in aligned_data:
        aligned_seqs.append(str(alignment_record.seq))
    return aligned_seqs

def generate_logo_plot(aligned_seq_list, output_file):
    count_matrix = logomaker.alignment_to_matrix(aligned_seq_list, to_type='counts')
    
    # Custom color scheme configuration
    color_scheme = {
        'A': 'green',
        'C': 'blue',
        'D': 'red',
        'E': 'red',
        'F': 'cyan',
        'G': 'orange',
        'H': 'yellow',
        'I': 'cyan',
        'K': 'magenta',
        'L': 'cyan',
        'M': 'cyan',
        'N': 'purple',
        'P': 'purple',
        'Q': 'purple',
        'R': 'magenta',
        'S': 'green',
        'T': 'green',
        'V': 'cyan',
        'W': 'cyan',
        'Y': 'cyan',
    }
    
    plt.figure(figsize=(90, 30))
    logo_plot = logomaker.Logo(count_matrix, color_scheme=color_scheme)
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    # Demonstration section
    random.seed(42)
    
    # Load input data
    input_df = pd.read_csv("data.csv")
    
    # Random subset selection
    sampled_data = input_df.sample(n=250)
    seq_records = generate_seq_records(sampled_data)
    alignment_result = perform_alignment(seq_records)
    processed_sequences = extract_aligned_sequences(alignment_result)
    generate_logo_plot(processed_sequences, "figure.png")
    print("Process completed successfully!")