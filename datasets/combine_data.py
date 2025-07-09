import pandas as pd
import glob
import os

# Get the directory where the script is located
script_dir = os.path.dirname(__file__) 

# Define paths relative to the script's directory
data_dir = os.path.join(script_dir, 'process_data') 
output_file = os.path.join(script_dir, 'combined_training_data.csv') 

# Find all CSV files in the directory
if not os.path.isdir(data_dir):
    print(f"Error: Data directory not found at {data_dir}")
else:
    all_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True) 

    if not all_files:
        print(f"No CSV files found recursively in {data_dir}")
    else:
        # Define the columns to keep, now including light chain data
        required_columns = [
            'vh', 'vl', 'Antigen Sequence', 
            'H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4',
            'L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4',
            'ANT_Binding'
        ]
        
        df_list = []
        for f in all_files:
            try:
                # Read the CSV and check for necessary columns before loading
                temp_df = pd.read_csv(f)
                # Ensure all required columns exist in the file
                if all(col in temp_df.columns for col in required_columns):
                    df_list.append(temp_df[required_columns])
                else:
                    # Find which columns are missing for a more informative message
                    missing_cols = [col for col in required_columns if col not in temp_df.columns]
                    print(f"Warning: Skipping file {f} because it lacks required columns: {missing_cols}")

            except Exception as e:
                print(f"Warning: Error processing file {f}: {e}. Skipping file.")

        if not df_list:
            print(f"No valid data found in CSV files in {data_dir} with all required columns.")
        else:
            # Concatenate all the dataframes in the list
            combined_df = pd.concat(df_list, ignore_index=True)

            # --- Data Validation ---
            # Drop rows where either heavy or light chain sequences are missing
            initial_rows = len(combined_df)
            combined_df.dropna(subset=['vh', 'vl'], inplace=True)
            validated_rows = len(combined_df)
            if initial_rows > validated_rows:
                print(f"Removed {initial_rows - validated_rows} rows with missing 'vh' or 'vl' sequences.")

            # Drop duplicate rows based on all columns
            initial_rows = len(combined_df)
            combined_df.drop_duplicates(inplace=True)
            final_rows = len(combined_df)
            if initial_rows > final_rows:
                print(f"Removed {initial_rows - final_rows} duplicate rows.")

            # Split into train and test (e.g., 80% train, 20% test)
            train_df = combined_df.sample(frac=0.8, random_state=42)
            test_df = combined_df.drop(train_df.index)

            # Save the splits
            train_file = os.path.join(script_dir, 'combined_training_data.csv')
            test_file = os.path.join(script_dir, 'test_data.csv')
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            print(f"Train data saved to {train_file}")
            print(f"Test data saved to {test_file}")