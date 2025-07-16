#!/usr/bin/env python3
"""
Antibody Sequence Splitting Script

This script splits antibody sequences into CDR and Framework regions using
the same professional tools (ANARCI/abnumber) as the main processing pipeline.

Usage:
    python antibody_split.py

Modify the INPUT_FILE and OUTPUT_FILE variables below to point to your data.

Input CSV Requirements:
    - vh: Heavy chain sequence
    - vl: Light chain sequence  
    - Antigen Sequence: Target protein sequence
    - Any other columns (will be preserved)

Output CSV will include all original columns plus:
    - H-FR1, H-CDR1, H-FR2, H-CDR2, H-FR3, H-CDR3, H-FR4: Heavy chain regions
    - L-FR1, L-CDR1, L-FR2, L-CDR2, L-FR3, L-CDR3, L-FR4: Light chain regions
"""

import sys
import os
import pandas as pd
from process_all_data import run_combined_chain_processing

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR DATA
# =============================================================================

INPUT_FILE = "path/to/your/input_file.csv"  # Change this to your input file
OUTPUT_FILE = "path/to/your/output_file.csv"  # Change this to your output file

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def split_antibody_file(input_path, output_path, scheme="chothia"):
    """
    Split antibody sequences in a CSV file into CDR/FR regions.
    
    Args:
        input_path (str): Path to input CSV file with vh, vl, Antigen Sequence columns
        output_path (str): Path to save the processed CSV file
        scheme (str): Numbering scheme to use (chothia, imgt, kabat)
    """
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        print("Please modify the INPUT_FILE variable at the top of this script.")
        return False
    
    # Check if output directory exists, create if not
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Processing antibody sequences from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Load the input data
        df = pd.read_csv(input_path)
        
        # Check required columns
        if 'vh' not in df.columns or 'vl' not in df.columns:
            print(f"Error: Input file must have 'vh' and 'vl' columns!")
            print(f"Found columns: {list(df.columns)}")
            return False
        
        print(f"Processing {len(df)} antibody sequences using {scheme} numbering scheme...")
        
        # Use the existing professional processing function
        run_combined_chain_processing(df, output_path, scheme)
        
        print(f"✅ Successfully processed antibody sequences!")
        print(f"📄 Output saved to: {output_path}")
        print("\nProcessed columns added:")
        print("  Heavy chain: H-FR1, H-CDR1, H-FR2, H-CDR2, H-FR3, H-CDR3, H-FR4")
        print("  Light chain: L-FR1, L-CDR1, L-FR2, L-CDR2, L-FR3, L-CDR3, L-FR4")
        return True
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("\nCommon issues:")
        print("- Make sure your CSV has 'vh', 'vl' columns")
        print("- Ensure ANARCI/abnumber is properly installed")
        print("- Check that the antibody sequences are valid protein sequences")
        return False

def main():
    """Main entry point"""
    
    print("🧬 AntiBinder Antibody Sequence Splitter")
    print("=" * 50)
    
    # Check if user has modified the configuration
    if INPUT_FILE == "path/to/your/input_file.csv":
        print("❌ Please modify the INPUT_FILE variable at the top of this script!")
        print("   Set it to the path of your CSV file with antibody sequences.")
        print("\nExample:")
        print('   INPUT_FILE = "my_antibodies.csv"')
        print('   OUTPUT_FILE = "my_antibodies_processed.csv"')
        return
    
    if OUTPUT_FILE == "path/to/your/output_file.csv":
        print("❌ Please modify the OUTPUT_FILE variable at the top of this script!")
        print("   Set it to where you want the processed results saved.")
        return
    
    # Process the file
    success = split_antibody_file(INPUT_FILE, OUTPUT_FILE)
    
    if success:
        print("\n🎉 Processing complete! Your antibody sequences have been split into CDR/FR regions.")
    else:
        print("\n💡 Need help? Check the README.md for troubleshooting tips.")

if __name__ == "__main__":
    main()
