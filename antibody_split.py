#!/usr/bin/env python3
"""
Use Existing Antibody Processing

This script demonstrates how to use the existing process_all_data.py functionality
to split antibody sequences into CDR/FR regions.
"""

import os
import sys
import pandas as pd
from process_all_data import run_combined_chain_processing

def split_antibody_file(input_csv, output_csv, scheme="chothia"):
    """
    Use the existing process_all_data functionality to split antibody sequences.
    
    Args:
        input_csv: Path to CSV with 'vh' and 'vl' columns
        output_csv: Path to save results with CDR/FR regions
        scheme: Numbering scheme to use (chothia, imgt, kabat)
    """
    
    # Load the input data
    df = pd.read_csv(input_csv)
    
    # Check required columns
    if 'vh' not in df.columns or 'vl' not in df.columns:
        print(f"Error: Input file must have 'vh' and 'vl' columns!")
        print(f"Found columns: {list(df.columns)}")
        return False
    
    print(f"Processing {len(df)} sequences using {scheme} numbering scheme...")
    
    # Use the existing processing function
    run_combined_chain_processing(df, output_csv, scheme)
    
    print(f"Results saved to {output_csv}")
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "predictions/Abtlas_vh_vl_antigen_test_input_1.csv"
    output_file = "predictions/properly_split_regions.csv"
    
    if split_antibody_file(input_file, output_file):
        # Load and show results
        result_df = pd.read_csv(output_file)
        print("\nSample results:")
        print("=" * 80)
        
        region_cols = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4',
                      'L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4']
        
        for i in range(min(2, len(result_df))):
            print(f"\nSequence {i+1}:")
            for col in region_cols:
                if col in result_df.columns:
                    seq = result_df.iloc[i][col]
                    print(f"  {col}: {seq} (len={len(seq)})")
