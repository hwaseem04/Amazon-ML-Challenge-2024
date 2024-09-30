import pandas as pd
from constants import allowed_units

def process_csv(input_file_name):
    
    df = pd.read_csv(input_file_name)

    # Split 'prediction' column into 'number' and 'unit' columns
    df[['number', 'unit']] = df['prediction'].str.extract(r'(\[?[0-9.]+(?:,\s*[0-9.]+)?\]?)\s*(\w.+)')
    # df[['number', 'unit']] = df['prediction'].str.extract(r'([0-9.]+)\s*(\w.+)')
    df['number'] = df['number'].apply(lambda x: str(x).split(',')[1].strip('[] ') if isinstance(x, str) and ',' in x else x)

    # Replace 'prediction', 'number', 'unit' with '' when the unit is not in the allowed units
    df.loc[~df['unit'].isin(allowed_units), ['prediction', 'number', 'unit']] = ''

    # Remove rows where 'number' has more than one decimal point
    df.loc[df['number'].str.count('\.') > 1, ['prediction', 'number', 'unit']] = ''

    # Merge 'number' and 'unit' into a new 'prediction' column
    df['prediction'] = df['number'].astype(str) + ' ' + df['unit'].astype(str)

    # Handle cases where 'number' or 'unit' is missing by stripping extra spaces
    df['prediction'] = df['prediction'].str.strip()

    # Remove the 'number' and 'unit' columns
    df = df.drop(columns=['number', 'unit'])

    # Define output file name
    output_file_name = 'test_out.csv'

    df.to_csv(output_file_name, index=False)

    print(f"Processed file saved as: {output_file_name}")

process_csv('reindex_pred.csv')
