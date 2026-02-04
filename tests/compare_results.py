
import pandas as pd
import numpy as np
import sys

try:
    old = pd.read_csv('tests/benchmark_baseline_old.csv', index_col=0)
    new = pd.read_csv('tests/benchmark_baseline.csv', index_col=0)

    # Compare pvalues
    # Floating point comparison
    diff = np.abs(old['pvalues'] - new['pvalues'])
    max_diff = diff.max()
    
    print(f"Max p-value difference: {max_diff}")
    
    if max_diff < 1e-6:
        print("PASS: Results match within tolerance.")
    else:
        print("FAIL: Results do not match.")
        print("Top differences:")
        diff_df = pd.DataFrame({'old': old['pvalues'], 'new': new['pvalues'], 'diff': diff})
        print(diff_df.sort_values('diff', ascending=False).head())

except FileNotFoundError:
    print("One of the benchmark files is missing.")
    sys.exit(1)
except Exception as e:
    print(f"Error comparing results: {e}")
    sys.exit(1)
