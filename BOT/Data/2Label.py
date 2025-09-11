import pandas as pd
import os

def preprocess_and_save_data(input_file_path, output_file_path):
    """
    Loads a CSV file, combines specific columns into a single 'text' column,
    renames the 'Label' column to 'label', and saves the result to a new CSV file.

    Args:
        input_file_path (str): The path to the original multi-column CSV file.
        output_file_path (str): The path where the new, preprocessed CSV will be saved.
    """
    try:
        print(f"Loading data from: {input_file_path}")
        df = pd.read_csv(input_file_path)

        # Define the columns to be combined into a single text string
        text_columns = ['Method', 'URL', 'HTTP_Version', 'User-Agent', 'Pragma',
                        'Cache-control', 'Accept', 'Accept-Encoding', 'Accept-Charset',
                        'Accept-Language', 'Host', 'Cookie', 'Connection', 'Content-Type',
                        'Content-Length']

        # Ensure all columns exist in the DataFrame
        if not all(col in df.columns for col in text_columns):
            missing_cols = [col for col in text_columns if col not in df.columns]
            raise ValueError(f"Input file is missing the following columns: {missing_cols}")

        # Combine the selected columns into a new 'text' column
        # .astype(str) handles any non-string values gracefully
        df['text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

        # Keep only the new 'text' column and the original 'Label' column
        df = df[['text', 'Label']]

        # Rename the 'Label' column to 'label' to match Hugging Face conventions
        df = df.rename(columns={'Label': 'label'})

        print(f"Saving preprocessed data to: {output_file_path}")
        # Save the resulting DataFrame to a new CSV file
        # index=False prevents pandas from writing the DataFrame index as a column
        df.to_csv(output_file_path, index=False)
        
        print("Data preprocessing complete!")

    except FileNotFoundError:
        print(f"Error: The file at {input_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the paths for the input and output files
    # The current directory is assumed to be the project root 'BOT'
    input_file = os.path.join("balanced_dataset.csv")
    output_file = os.path.join("equal_dataset.csv")
    
    # Run the function with the defined paths
    preprocess_and_save_data(input_file, output_file)