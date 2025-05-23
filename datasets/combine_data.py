import pandas as pd
import glob
import os

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Define paths relative to the script's directory
# Assumes 'process_data' is a sibling folder to where the script is (or adjust as needed)
# If the script is IN the 'datasets' folder, and 'process_data' is also IN 'datasets':
data_dir = os.path.join(script_dir, 'process_data') 
output_file = os.path.join(script_dir, 'combined_training_data.csv') 

# Find all CSV files in the directory
# Make sure the data_dir exists before using glob
if not os.path.isdir(data_dir):
    print(f"Error: Data directory not found at {data_dir}")
else:
    all_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True) 

    if not all_files: # Now check if the list is empty
        print(f"No CSV files found recursively in {data_dir}")
    else:
        # Define the columns to keep - UPDATED LIST
        required_columns = ['vh', 'Antigen Sequence', 'H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4', 'ANT_Binding']
        
        df_list = []
        for f in all_files:
            try:
                # Read the CSV and select only the required columns
                df = pd.read_csv(f, usecols=required_columns)
                df_list.append(df)
            except ValueError as e:
                print(f"Warning: Could not read required columns from {f}. Error: {e}. Skipping file.")
            except Exception as e:
                print(f"Warning: Error reading file {f}: {e}. Skipping file.")
        if not df_list:
                print(f"No valid data found in CSV files in {data_dir}")
        else:
            # Concatenate all the dataframes in the list
            combined_df = pd.concat(df_list, ignore_index=True)

            # Drop duplicate rows based on all columns
            initial_rows = len(combined_df)
            combined_df.drop_duplicates(inplace=True)
            final_rows = len(combined_df)
            print(f"Removed {initial_rows - final_rows} duplicate rows.")


            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True) 
            
            # Save the combined data with only the required columns
            combined_df.to_csv(output_file, index=False)

            print(f"Combined data with selected columns saved to {output_file}")
