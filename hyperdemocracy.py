from datasets import load_dataset
import pandas as pd

def load_assembly_records() -> pd.DataFrame: 
    ds = load_dataset("hacdc/hyperdemocracy", split="train")
    return ds.to_pandas()

