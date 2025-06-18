import requests
from tqdm import tqdm
import zipfile
import os
import polars as pl


# Source: https://archive.ics.uci.edu/dataset/280/higgs

def main():
    # Streamed download with progress bar
    url = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'
    response = requests.get(url, stream=True)
    
    # 2.63 GB in bytes = 2.63 * 1024^3 
    default_size_bytes = int(2.63 * 1024 ** 3)
    total_size = int(response.headers.get('content-length', default_size_bytes))
    block_size = 1024  

    with open('higgs.zip', 'wb') as file, tqdm(
        desc='Downloading',
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    # Extract gz file contents
    os.makedirs('./higgs_data', exist_ok=True)
    
    with zipfile.ZipFile('higgs.zip', 'r') as archive:
        
        gz_filename = next(f for f in archive.namelist() if f.endswith('.gz'))
        
        with archive.open(gz_filename) as gz_file:            
            df = pl.read_csv(gz_file, has_header=False)
        
    # Rename columns
    columns = [ # column names described in the source page
        'target', 'lepton pT', 'lepton eta', 'lepton phi',
        'missing energy magnitude', 'missing energy phi',
        'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag',
        'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag',
        'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag',
        'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
    
    df = df.rename(dict(zip(df.columns, columns)))
        
    # Save as feather format
    df.write_parquet('./higgs_data/HIGGS.parquet')
    
    # Clean-up
    os.remove('higgs.zip')
    

    print('Download and extraction complete.')

if __name__ == '__main__':
    main()