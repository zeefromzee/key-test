# Cryptographic Key Analysis Tool

A comprehensive Python tool that performs statistical and cryptographic tests on cryptographic keys. This tool implements the complete NIST Statistical Test Suite (15 tests) and additional advanced randomness tests (16+ tests) to evaluate the quality and security of cryptographic keys.

**Note:** Some tests require minimum bit lengths (e.g., 1024+ bits for matrix rank test, 387,840 bits for Maurer's Universal test). Tests that cannot run due to insufficient bits will be automatically SKIPPED and won't count against your key's pass/fail rate.

## Features

- **Complete NIST SP 800-22 Statistical Test Suite** 
  - Frequency (Monobit) Test
  - Block Frequency Test
  - Runs Test
  - Longest Run of Ones Test
  - Binary Matrix Rank Test
  - Discrete Fourier Transform (Spectral) Test
  - Non-overlapping Template Matching Test
  - Overlapping Template Matching Test
  - Maurer's Universal Statistical Test
  - Linear Complexity Test
  - Serial Test
  - Approximate Entropy Test
  - Cumulative Sums Test
  - Random Excursions Test
  - Random Excursions Variant Test

- **Additional Cryptographic Randomness Tests** 
  - Autocorrelation Test
  - Shannon Entropy Test
  - Chi-Square Uniformity Test
  - Poker Test
  - Gap Test
  - Collision Test
  - Birthday Spacing Test
  - Bit Independence Test
  - Avalanche Effect Test
  - Strict Avalanche Criterion
  - Run Distribution Test
  - Coupon Collector's Test
  - Permutation Test
  - Periodicity Detection Test
  - Extended Maurer's Universal Test
  - Lempel-Ziv Complexity Test

- **Key Quality Metrics**
  - Key Strength Estimation
  - Uniformity Analysis
  - Complexity Score

## Quick Start Guide

### 1. Prepare Your CSV File

Create a CSV file with your cryptographic keys, one key per row:

```csv
3a4f2e8b9c1d7f6a5e3b8c9d2f1a4e7b3c6d9f2a5e8b1c4d7f9a2e5b8c1d3f6a
7f2a5e8b1c4d7f9a2e5b8c1d3f6a9c2e5b8f1d4a7c9e2b5f8d1a4c7e9b2f5a8d1
9c2e5b8f1d4a7c9e2b5f8d1a4c7e9b2f5a8d1c4e7b9f2a5d8c1e4b7a9f2c5e8b1
```

**Supported key formats:**
- `hex` - Hexadecimal (default)
- `binary` - Binary string (e.g., "10110101...")
- `base64` - Base64 encoded
- `decimal` - Decimal number

### 2. Configure the Tool

Open `crypto_key_analyzer.py` and modify the **CONFIGURATION** section at the top:

```python
CONFIG = {
    # Input file settings
    'csv_file': 'keys.csv',              # ← Change this to your CSV file path
    'key_column': 0,                      # ← Column index (0 = first column)
    'key_encoding': 'hex',                # ← Change if using different format
    
    # Test parameters
    'significance_level': 0.01,           # ← Alpha level (0.01 = 99% confidence)
    'min_bit_length': 128,                # ← Minimum bits required
    
    # Output settings
    'output_dir': 'crypto_analysis_results',
    'generate_plots': True,               # ← Set False to skip plots
    'save_csv_report': True,              # ← Set False to skip CSV reports
    'verbose': True,                       # ← Set False for less output
}
```

### 3. Run the Analysis

```bash
python crypto_key_analyzer.py
```

### 4. Review Results

The tool generates:
- **Console output**: Summary and detailed test results
- **CSV reports**: `crypto_analysis_results/key_X_detailed_report.csv`
- **Plots**: `crypto_analysis_results/key_X_analysis.png`
- **Overall summary**: `crypto_analysis_results/overall_summary.csv`

## Understanding the Results

### Test Results

Each test produces a **p-value** between 0 and 1:
- **p-value ≥ 0.01** (default): PASS ✓ - Key appears random
- **p-value < 0.01**: FAIL ✗ - Key shows non-random patterns
- **SKIP** ⊘ - Test requires more bits than available (doesn't count as pass or fail)

### Key Quality Scores

- **Key Strength**: 0-100% (higher is better, aim for >70%)
- **Uniformity**: 0-100% (higher is better, aim for >70%)
- **Complexity**: 0-100% (higher is better, aim for >70%)

### What Does a FAIL Mean?

A failed test indicates the key exhibits patterns that deviate from true randomness. This could mean:
- Weak key generation algorithm
- Insufficient entropy source
- Potential security vulnerability

**Recommendation**: Keys that fail multiple tests should not be used for cryptographic purposes.

## Customization Guide

### Change Input File Format

If your keys are in binary format:
```python
'key_encoding': 'binary',
```

If your keys are in a different column:
```python
'key_column': 2,  # Third column (0-indexed)
```

### Adjust Test Sensitivity

For stricter testing (99.9% confidence):
```python
'significance_level': 0.001,
```

For more lenient testing (95% confidence):
```python
'significance_level': 0.05,
```

### Modify Test Parameters

```python
'block_size': 256,              # Larger blocks for longer keys
'template_size': 12,            # Longer templates for pattern matching
'serial_block_length': 32,      # Larger blocks for serial test
```

### Disable Plots (Faster Analysis)

```python
'generate_plots': False,
```

## Example Output

```
╔════════════════════════════════════════════════════════════════════════╗
║         CRYPTOGRAPHIC KEY ANALYSIS TOOL                                ║
║          Statistical & Cryptographic Tests                             ║
╚════════════════════════════════════════════════════════════════════════╝

Reading keys from: keys.csv
Found 3 key(s) to analyze

================================================================================
CRYPTOGRAPHIC KEY ANALYSIS SUMMARY
================================================================================
Key ID: KEY_1
Bit Length: 256
Total Tests: 34
Tests Run: 29
Tests Skipped: 5 (insufficient bits for these tests)
Tests Passed: 19 (65.52% of applicable tests)
Tests Failed: 10 (34.48% of applicable tests)
Significance Level: 0.01
================================================================================
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Pandas
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy pandas matplotlib
```

## Tips for Best Results

1. **Use sufficient key length**: Minimum 128 bits recommended, 256+ bits ideal
2. **Test multiple keys**: Analyze several keys to identify systemic issues
3. **Review all failures**: Even one failed test can indicate problems
4. **Compare results**: Test known-good keys as a baseline
5. **Update configurations**: Adjust test parameters based on your key length

## Troubleshooting

**"No keys found in CSV file"**
- Check the file path in CONFIG
- Verify the CSV file exists
- Ensure keys are in the correct column

**"Key has only X bits (minimum 128 required)"**
- Your key is too short for reliable analysis
- Reduce `min_bit_length` or use longer keys

**"Error converting key to binary"**
- Check that `key_encoding` matches your key format
- Verify keys are valid hex/binary/base64

## Advanced Usage

### Analyzing Different Key Formats

**Binary keys:**
```python
CONFIG = {
    'csv_file': 'binary_keys.csv',
    'key_encoding': 'binary',
    ...
}
```

**Base64 keys:**
```python
CONFIG = {
    'csv_file': 'base64_keys.csv',
    'key_encoding': 'base64',
    ...
}
```

### Batch Analysis

Place all your keys in one CSV file (one per row), and the tool will analyze them all sequentially.

## Security Notice

This tool is for **analysis only**. It does not:
- Generate cryptographic keys
- Store or transmit keys
- Modify your keys

**Always protect your cryptographic keys.** Do not share the CSV files containing real production keys.

## License

This tool is provided as-is for cryptographic analysis purposes.

## Support

For issues or questions:
1. Check that all configuration settings are correct
2. Verify your CSV file format
3. Ensure all dependencies are installed
4. Review the console output for specific error messages
