# Cryptographic Key Analysis Tool

**A Comprehensive Statistical & Cryptographic Randomness Test Suite**

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![NIST SP 800-22](https://img.shields.io/badge/NIST-SP%20800--22-green.svg)](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)
[![Test Suite](https://img.shields.io/badge/tests-34-brightgreen.svg)](#test-suite-details)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start Guide](#quick-start-guide)
- [Configuration](#configuration)
- [Understanding Results](#understanding-results)
- [Test Suite Details](#test-suite-details)
- [Customization Guide](#customization-guide)
- [Requirements](#requirements)
- [Tips for Best Results](#tips-for-best-results)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Security Notice](#security-notice)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Author](#author)
- [Version History](#version-history)

---

## Overview

The **Cryptographic Key Analysis Tool** is a comprehensive Python-based testing suite that performs **34 statistical and cryptographic randomness tests** on cryptographic keys. This tool implements the complete NIST Statistical Test Suite (SP 800-22 Rev. 1a) consisting of 15 tests, plus an additional 16 advanced randomness tests to thoroughly evaluate the quality and security of cryptographic key material.

The tool provides detailed analysis including p-values, pass/fail determinations, key quality metrics, and comprehensive reporting capabilities to help security professionals and researchers validate the randomness properties of their cryptographic keys.

**Note:** Some tests require minimum bit lengths (e.g., 1024+ bits for matrix rank test, 387,840 bits for Maurer's Universal test). Tests that cannot run due to insufficient bits will be automatically skipped and will not count against your key's pass/fail rate.

---

## Features

### Complete NIST SP 800-22 Statistical Test Suite (15 tests)

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

### Additional Cryptographic Randomness Tests (16 tests)

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

### Key Quality Metrics

- Key Strength Estimation
- Uniformity Analysis
- Complexity Score

---

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
    'csv_file': 'keys.csv',              # Change this to your CSV file path
    'key_column': 0,                      # Column index (0 = first column)
    'key_encoding': 'hex',                # Change if using different format
    
    # Test parameters
    'significance_level': 0.01,           # Alpha level (0.01 = 99% confidence)
    'min_bit_length': 128,                # Minimum bits required
    
    # Output settings
    'output_dir': 'crypto_analysis_results',
    'generate_plots': True,               # Set False to skip plots
    'save_csv_report': True,              # Set False to skip CSV reports
    'verbose': True,                       # Set False for less output
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

---

## Configuration

### Input File Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `csv_file` | Path to CSV file containing keys | `keys.csv` |
| `key_column` | Column index for keys (0-based) | `0` |
| `key_encoding` | Key format: `hex`, `binary`, `base64`, `decimal` | `hex` |

### Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `significance_level` | Alpha level for statistical tests | `0.01` |
| `min_bit_length` | Minimum bits required for analysis | `128` |
| `block_size` | Block size for frequency tests | `256` |
| `template_size` | Template length for pattern matching | `12` |
| `serial_block_length` | Block length for serial test | `32` |

### Output Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Directory for results and reports | `crypto_analysis_results` |
| `generate_plots` | Generate visualization plots | `True` |
| `save_csv_report` | Save detailed CSV reports | `True` |
| `verbose` | Display detailed console output | `True` |

---

## Understanding Results

### Test Results

Each test produces a **p-value** between 0 and 1:

- **p-value ≥ 0.01** (default): **PASS** — Sequence appears random
- **p-value < 0.01**: **FAIL** — Sequence exhibits non-random patterns
- **SKIP**: Test requires more bits than available (doesn't count as pass or fail)

### Key Quality Scores

| Metric | Range | Target | Description |
|--------|-------|--------|-------------|
| **Key Strength** | 0-100% | >70% | Overall randomness quality |
| **Uniformity** | 0-100% | >70% | Bit distribution balance |
| **Complexity** | 0-100% | >70% | Pattern complexity measure |

### What Does a FAIL Mean?

A failed test indicates the key exhibits patterns that deviate from true randomness. This could mean:

- Weak key generation algorithm
- Insufficient entropy source
- Potential security vulnerability

**Recommendation:** Keys that fail multiple tests should not be used for cryptographic purposes.

---

## Test Suite Details

### NIST SP 800-22 Statistical Test Suite

The tool implements all 15 tests from the NIST Special Publication 800-22 Rev. 1a:

| Test Name | Purpose | Minimum Bits |
|-----------|---------|--------------|
| **Frequency (Monobit)** | Tests proportion of zeros and ones | 100 |
| **Block Frequency** | Tests frequency within M-bit blocks | 128 |
| **Runs** | Tests oscillation between zeros and ones | 100 |
| **Longest Run of Ones** | Tests longest run of ones in M-bit blocks | 128 |
| **Binary Matrix Rank** | Tests linear dependence | 1024 |
| **Spectral (DFT)** | Tests periodic features | 1000 |
| **Non-overlapping Template** | Tests occurrence of pre-specified patterns | 1000 |
| **Overlapping Template** | Tests number of occurrences of patterns | 1000 |
| **Maurer's Universal** | Tests compressibility | 387,840 |
| **Linear Complexity** | Tests sequence complexity | 1,000,000 |
| **Serial** | Tests frequency of all patterns | 128 |
| **Approximate Entropy** | Tests frequency of overlapping blocks | 100 |
| **Cumulative Sums** | Tests cumulative sum deviations | 100 |
| **Random Excursions** | Tests state visits in random walk | 1,000,000 |
| **Random Excursions Variant** | Tests deviations in random walk | 1,000,000 |

### Additional Cryptographic Tests

These 16 supplementary tests provide comprehensive coverage beyond NIST requirements:

- **Autocorrelation**: Detects correlation between bit sequences
- **Shannon Entropy**: Measures information content
- **Chi-Square**: Tests uniformity of bit distribution
- **Poker**: Tests distribution of m-bit patterns
- **Gap**: Tests spacing between occurrences of patterns
- **Collision**: Tests for duplicate sequences
- **Birthday Spacing**: Tests spacing between matching patterns
- **Bit Independence**: Tests statistical independence of bits
- **Avalanche Effect**: Tests diffusion properties
- **Strict Avalanche Criterion**: Tests bit change propagation
- **Run Distribution**: Tests distribution of run lengths
- **Coupon Collector**: Tests time to see all patterns
- **Permutation**: Tests order of subsequences
- **Periodicity Detection**: Tests for cyclic patterns
- **Extended Universal**: Enhanced compressibility test
- **Lempel-Ziv Complexity**: Tests algorithmic complexity

---

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
'significance_level': 0.001,  # Super strict
```

For more lenient testing (95% confidence):
```python
'significance_level': 0.05,  # More relaxed
```

### Modify Test Parameters

```python
'block_size': 256,              # Larger blocks for longer keys
'template_size': 12,            # Longer templates for pattern matching
'serial_block_length': 32,      # Larger blocks for serial test
```

### Disable Plots (Faster Analysis)

```python
'generate_plots': False,  # Skip visualization generation
```

---

## Example Output

```
╔════════════════════════════════════════════════════════════════════════╗
║         CRYPTOGRAPHIC KEY ANALYSIS TOOL                                ║
║         Statistical & Cryptographic Tests                              ║
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

---

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

---

## Tips for Best Results

1. **Use sufficient key length**: Minimum 128 bits recommended, 256+ bits ideal
2. **Test multiple keys**: Analyze several keys to identify systemic issues
3. **Review all failures**: Even one failed test can indicate problems
4. **Compare results**: Test known-good keys as a baseline
5. **Update configurations**: Adjust test parameters based on your key length

---

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

---

## File Structure

```
crypto-key-analyzer/
│
├── crypto_key_analyzer.py    # Main analysis tool
├── README.md                  # This file
├── USAGE_GUIDE.txt            # Detailed usage instructions
│
├── keys.csv                   # Input: cryptographic keys (user-provided)
│
└── crypto_analysis_results/   # Output directory (auto-created)
    ├── key_1_detailed_report.csv
    ├── key_1_analysis.png
    ├── key_2_detailed_report.csv
    ├── key_2_analysis.png
    └── overall_summary.csv
```

---

## Security Notice

This tool is for **analysis only**. It does not:
- Generate cryptographic keys
- Store or transmit keys
- Modify your keys

**Always protect your cryptographic keys.** Do not share the CSV files containing real production keys.

---

## Contributing

Contributions that improve the accuracy, performance, or functionality of this tool are welcome.

### Contribution Guidelines

1. **Fork the Repository** and create a feature branch
2. **Follow Coding Standards**: PEP 8 for Python, clear documentation
3. **Add Tests**: Include test cases for new features
4. **Submit Pull Request**: Provide detailed description of changes

### Areas for Contribution

- Additional statistical tests
- Performance optimizations
- Enhanced visualization options
- Cross-platform compatibility improvements
- Comprehensive documentation improvements

---

## Citation

If you use this software in academic research, technical reports, or publications, please cite as follows:

### APA Format
```
zeefromzee. (2026). Cryptographic Key Analysis Tool [Computer software]. 
GitHub. https://github.com/zeefromzee/key-test
```

### BibTeX Format
```bibtex
@software{crypto_key_analyzer_2026,
  author = {zeefromzee},
  title = {Cryptographic Key Analysis Tool},
  year = {2026},
  url = {https://github.com/zeefromzee/key-test},
  note = {34-test statistical and cryptographic randomness test suite}
}
```

### IEEE Format
```
zeefromzee, "Cryptographic Key Analysis Tool," GitHub repository, 2026. 
[Online]. Available: https://github.com/zeefromzee/key-test
```

---

## License

This tool is provided as-is for cryptographic analysis purposes.

Copyright © 2026 zeefromzee. All Rights Reserved.

---

## Author

**GitHub**: [@zeefromzee](https://github.com/zeefromzee)  
**Repository**: [key-test](https://github.com/zeefromzee/key-test)

For questions, bug reports, or collaboration inquiries, please open an issue on GitHub.

---

## Version History

### v1.0.0 (Current)
- Initial release
- Complete NIST SP 800-22 Statistical Test Suite (15 tests)
- Additional cryptographic randomness tests (16 tests)
- Key quality metrics (strength, uniformity, complexity)
- CSV input/output support
- Visualization and plotting capabilities
- Comprehensive reporting

---

**Last Updated**: February 2026  
**License**: MIT  
**Status**: Production Ready for Research & Analysis

---

*This software is provided for educational and research purposes. The tool performs statistical analysis of cryptographic keys and does not guarantee security. Users should conduct independent security analysis before deploying keys in production environments.*
