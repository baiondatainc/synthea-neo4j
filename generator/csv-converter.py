#!/usr/bin/env python3
"""
Convert parquet files to CSV format.
Reads all .parquet files from the input directory and writes them as CSV files
to the input/csv directory.
"""

import os
import sys
from pathlib import Path
import pandas as pd

def convert_parquet_to_csv(input_dir: str, output_dir: str) -> None:
    """
    Convert all parquet files in input_dir to CSV files in output_dir.
    
    Args:
        input_dir: Directory containing .parquet files
        output_dir: Directory where CSV files will be written
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet file(s) to convert\n")
    
    for parquet_file in sorted(parquet_files):
        csv_file = output_path / f"{parquet_file.stem}.csv"
        
        try:
            print(f"Converting: {parquet_file.name} → {csv_file.name}")
            
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            # Write to CSV
            df.to_csv(csv_file, index=False)
            
            print(f"  ✓ Success ({len(df)} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            sys.exit(1)
    
    print(f"\n✓ All files converted successfully to {output_dir}")

if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    output_dir = script_dir / "input" / "csv"
    
    # Run conversion
    convert_parquet_to_csv(str(input_dir), str(output_dir))
