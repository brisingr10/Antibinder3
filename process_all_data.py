import os
import pandas as pd
from abnumber import Chain
from joblib import Parallel, delayed
from tqdm import tqdm
import glob

def process_sequence(seq_index, sequence, scheme, chain_type_expected):
    output = {'Seq_Index': seq_index}
    try:
        ab_chain = Chain(sequence, scheme=scheme, use_anarcii=True)
        if ab_chain.chain_type == chain_type_expected or (chain_type_expected == 'L' and ab_chain.chain_type in ['L', 'K']):
            prefix = 'H' if chain_type_expected == 'H' else 'L'
            output.update({
                f'{prefix}-FR1': ab_chain.fr1_seq or '',
                f'{prefix}-CDR1': ab_chain.cdr1_seq or '',
                f'{prefix}-FR2': ab_chain.fr2_seq or '',
                f'{prefix}-CDR2': ab_chain.cdr2_seq or '',
                f'{prefix}-FR3': ab_chain.fr3_seq or '',
                f'{prefix}-CDR3': ab_chain.cdr3_seq or '',
                f'{prefix}-FR4': ab_chain.fr4_seq or ''
            })
        else:
            print(f"Warning: Chain at index {seq_index} is not {chain_type_expected} chain. Found {ab_chain.chain_type}.")
    except Exception as error:
        print(f"Error processing index {seq_index}: {error}")
        prefix = 'H' if chain_type_expected == 'H' else 'L'
        output.update({
            f'{prefix}-FR1': '', f'{prefix}-CDR1': '',
            f'{prefix}-FR2': '', f'{prefix}-CDR2': '',
            f'{prefix}-FR3': '', f'{prefix}-CDR3': '',
            f'{prefix}-FR4': ''
        })
    return output

def run_combined_chain_processing(dataframe, output_filepath, scheme_type):
    """Process both heavy and light chains and combine results in a single dataframe"""
    
    # Process Heavy Chain
    print("Processing heavy chain sequences...")
    parallel_processor = Parallel(n_jobs=-1, backend="loky")
    heavy_processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type, 'H') 
        for idx, sequence in tqdm(dataframe['vh'].dropna().items(), desc="Processing Heavy Chain Sequences")
    )
    heavy_processed_df = pd.DataFrame(heavy_processed_list).set_index('Seq_Index')
    
    # Process Light Chain
    print("Processing light chain sequences...")
    light_processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type, 'L') 
        for idx, sequence in tqdm(dataframe['vl'].dropna().items(), desc="Processing Light Chain Sequences")
    )
    light_processed_df = pd.DataFrame(light_processed_list).set_index('Seq_Index')
    
    # Start with the original dataframe
    combined_df = dataframe.copy()
    
    # Add heavy chain columns
    for column in heavy_processed_df.columns:
        combined_df[column] = heavy_processed_df[column]
    
    # Add light chain columns
    for column in light_processed_df.columns:
        combined_df[column] = light_processed_df[column]
    
    # Save the combined result
    combined_df.to_csv(output_filepath, index=False)
    print(f"Combined processed data saved to {output_filepath}")

def run_chain_processing(dataframe, output_filepath, scheme_type, chain_col_name, chain_type_expected):
    parallel_processor = Parallel(n_jobs=-1, backend="loky")
    processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type, chain_type_expected) 
        for idx, sequence in tqdm(dataframe[chain_col_name].dropna().items(), desc=f"Processing {chain_type_expected} Chain Sequences")
    )
    processed_dataframe = pd.DataFrame(processed_list).set_index('Seq_Index')

    for column in processed_dataframe.columns:
        dataframe[column] = processed_dataframe[column]

    dataframe.to_csv(output_filepath, index=False)

def preprocess_and_split_raw_data(input_filepath, output_dir, heavy_col, light_col, antigen_col, binding_col, delta_g_cutoff=None):
    df = pd.read_csv(input_filepath)
    
    # Handle delta_g conversion if specified
    if binding_col == "delta_g" and delta_g_cutoff is not None:
        if "delta_g" not in df.columns:
            print(f"Error: 'delta_g' column not found in {input_filepath} for delta_g conversion. Skipping file.")
            return
        df['ANT_Binding'] = (df["delta_g"] <= delta_g_cutoff).astype(int)
        binding_col_final = 'ANT_Binding'
    else:
        binding_col_final = binding_col

    # Ensure required columns are present for final combination
    required_for_combine = [heavy_col, light_col, antigen_col, binding_col_final]
    for col in required_for_combine:
        if col not in df.columns:
            print(f"Error: Missing expected column '{col}' in {input_filepath}. Skipping file.")
            return

    # Rename columns for consistency with combine_data.py expectations
    df_processed = df.rename(columns={
        heavy_col: 'vh',
        light_col: 'vl',
        antigen_col: 'Antigen Sequence',
        binding_col_final: 'ANT_Binding'
    })

    # Process both Heavy and Light chains and combine in single file
    output_filepath = os.path.join(output_dir, os.path.basename(input_filepath).replace('.csv', '_processed.csv'))
    print(f"Processing both heavy and light chains for {input_filepath}...")
    run_combined_chain_processing(df_processed, output_filepath, "chothia")

def combine_and_split_training_data(data_dir, output_base_dir):
    all_files = glob.glob(os.path.join(data_dir, '*_processed.csv'))  # Look for _processed.csv files

    if not all_files:
        print(f"No processed CSV files found in {data_dir}")
        return

    required_columns = [
        'vh', 'vl', 'Antigen Sequence', 
        'H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4',
        'L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4',
        'ANT_Binding'
    ]
    
    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            if all(col in temp_df.columns for col in required_columns):
                df_list.append(temp_df[required_columns])
                print(f"Successfully loaded {len(temp_df)} rows from {f}")
            else:
                missing_cols = [col for col in required_columns if col not in temp_df.columns]
                print(f"Warning: Skipping file {f} because it lacks required columns: {missing_cols}")

        except Exception as e:
            print(f"Warning: Error processing file {f}: {e}. Skipping file.")

    if not df_list:
        print(f"No valid data found in processed CSV files in {data_dir} with all required columns.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined {len(combined_df)} total rows from {len(df_list)} files")

    initial_rows = len(combined_df)
    combined_df.dropna(subset=['vh', 'vl'], inplace=True)
    validated_rows = len(combined_df)
    if initial_rows > validated_rows:
        print(f"Removed {initial_rows - validated_rows} rows with missing 'vh' or 'vl' sequences.")

    initial_rows = len(combined_df)
    combined_df.drop_duplicates(inplace=True)
    final_rows = len(combined_df)
    if initial_rows > final_rows:
        print(f"Removed {initial_rows - final_rows} duplicate rows.")

    train_df = combined_df.sample(frac=0.8, random_state=42)
    test_df = combined_df.drop(train_df.index)

    train_file = os.path.join(output_base_dir, 'training_data.csv')
    test_file = os.path.join(output_base_dir, 'test_data.csv')
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Train data saved to {train_file} ({len(train_df)} rows)")
    print(f"Test data saved to {test_file} ({len(test_df)} rows)")

if __name__ == "__main__":
    base_raw_data_path = "./datasets/raw_data"
    processed_split_data_output_path = "./datasets/process_data"
    final_training_data_output_path = "./datasets"

    os.makedirs(processed_split_data_output_path, exist_ok=True)

    # Process CoV-AbDab_only_sars2_filter.csv
    preprocess_and_split_raw_data(
        input_filepath=os.path.join(base_raw_data_path, "COVID-19", "CoV-AbDab_only_sars2_filter.csv"),
        output_dir=processed_split_data_output_path,
        heavy_col="Heavy",
        light_col="Light",
        antigen_col="antigen",
        binding_col="Label"
    )

    # Process biomap.csv
    preprocess_and_split_raw_data(
        input_filepath=os.path.join(base_raw_data_path, "BioMap", "biomap.csv"),
        output_dir=processed_split_data_output_path,
        heavy_col="antibody_seq_a",
        light_col="antibody_seq_b",
        antigen_col="antigen_seq",
        binding_col="delta_g",
        delta_g_cutoff=-10.90 # Median value for ~50% binding
    )

    # Process met_a.csv
    # preprocess_and_split_raw_data(
    #     input_filepath=os.path.join(base_raw_data_path, "MET", "met_a.csv"),
    #     output_dir=processed_split_data_output_path,
    #     heavy_col="vh",
    #     light_col=None, # No light chain in this file
    #     antigen_col="Antigen Sequence",
    #     binding_col="ANT_Binding"
    # )

    # Process HIV dataset (commented out - needs special handling for single antibody column)
    # preprocess_and_split_raw_data(
    #     input_filepath=os.path.join(base_raw_data_path, "HIV", "dataset_hiv_cls.csv"),
    #     output_dir=processed_split_data_output_path,
    #     heavy_col="antibody", # Single antibody sequence - needs splitting
    #     light_col=None, # Would need to be extracted from antibody column
    #     antigen_col="virus_seq",
    #     binding_col="ANT_Binding"
    # )

    print("All raw data processed and split. Now combining into training/test datasets...")
    combine_and_split_training_data(processed_split_data_output_path, final_training_data_output_path)

    print("Data processing complete. Check 'datasets/training_data.csv' and 'datasets/test_data.csv'.")