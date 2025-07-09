import os
import pandas as pd
from abnumber import Chain
from joblib import Parallel, delayed
from tqdm import tqdm

def process_sequence(seq_index, sequence, scheme):
    output = {'Seq_Index': seq_index}
    try:
        print(f"Processing index {seq_index}: {sequence}")
        ab_chain = Chain(sequence, scheme=scheme, use_anarcii=True)
        print(f"Chain type: {ab_chain.chain_type}")
        if ab_chain.chain_type in ['L', 'K']:  # Check for light chain types (Lambda or Kappa)
            output.update({
                'L-FR1': ab_chain.fr1_seq or '',
                'L-CDR1': ab_chain.cdr1_seq or '',
                'L-FR2': ab_chain.fr2_seq or '',
                'L-CDR2': ab_chain.cdr2_seq or '',
                'L-FR3': ab_chain.fr3_seq or '',
                'L-CDR3': ab_chain.cdr3_seq or '',
                'L-FR4': ab_chain.fr4_seq or ''
            })
        else:
            print(f"Warning: Chain at index {seq_index} is not a light chain (L or K).")
    except Exception as error:
        print(f"Error processing index {seq_index}: {error}")
        output.update({
            'L-FR1': '', 'L-CDR1': '',
            'L-FR2': '', 'L-CDR2': '',
            'L-FR3': '', 'L-CDR3': '',
            'L-FR4': ''
        })
    return output

def process_file(input_filepath, output_filepath, scheme_type):
    dataframe = pd.read_csv(input_filepath)
    parallel_processor = Parallel(n_jobs=-1, backend="loky")
    processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type) 
        for idx, sequence in tqdm(dataframe['vl'].dropna().items(), desc="Processing Light Chain Sequences") # Changed to 'vl'
    )
    processed_dataframe = pd.DataFrame(processed_list).set_index('Seq_Index')

    for column in processed_dataframe.columns:
        dataframe[column] = processed_dataframe[column]

    dataframe.to_csv(output_filepath, index=False)

def run_processing(input_path, output_path, scheme):
    process_file(input_path, output_path, scheme)


if __name__ == "__main__":
    working_directory = "./"
    input_file_path = "predictions/input/met_a.csv"  # Adjust this path as needed
    output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + "_light_chain_processed.csv" # Changed output file name
    output_file_path = os.path.join(os.path.dirname(input_file_path), output_file_name)
    
    os.chdir(working_directory)

    primary_scheme = 'chothia' # Schemes like 'chothia' and 'kabat' work for light chains too
    backup_scheme = 'kabat' # Using kabat as a backup scheme

    run_processing(input_file_path, output_file_path, primary_scheme)
    print("Light Chain Processing Completed!")