# quantum-key-tests ✧*｡٩(ˊᗜˋ*)و✧*｡

Comprehensive test suite for quantum-inspired cryptographic key generation with NIST SP 800-22 statistical validation!

Perfect for testing your random keys, learning about entropy extraction, and making sure your crypto is actually crypto (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧

---

## What's Inside? ⋆｡˚ ☁︎ ˚｡⋆

### Core Components ✮⋆˙

**Von Neumann Debiasing** ໒꒰ྀི ˶• ˕ •˶ ꒱ྀིა  
Takes biased random bits and makes them perfectly balanced!
- `01` pairs → output `0`
- `10` pairs → output `1`  
- `00` and `11` → discarded (they're sus)

**Entropy Pooling** (˶ᵔ ᵕ ᵔ˶)  
XORs multiple entropy sources together for maximum randomness power!

**Multi-Stage Extraction** ⸜(｡˃ ᵕ ˂ )⸝  
- High-quality: SHA-512 based extraction
- Ultra-quality: Multiple rounds for extra paranoid security

**16 NIST Statistical Tests** ☆ﾟ.*･｡ﾟ  
All the tests from NIST SP 800-22 to validate your keys aren't predictable!

---

## Quick Start (ง •_•)ง

```bash
# Install dependencies
pip install pytest numpy pandas scipy hypothesis

# Run all tests
pytest test_quantum_key_generator.py -v

# Run with benchmarks
pytest test_quantum_key_generator.py -v --benchmark-enable

# Run specific test class
pytest test_quantum_key_generator.py::TestVonNeumannDebiasing -v
```

---

## Test Coverage ･ﾟ( ﾉд`ﾟ)

### Von Neumann Debiasing Tests ✧˖°
- Verifies 01→0 and 10→1 conversions
- Checks 00/11 pair discarding
- Deterministic output validation
- Size reduction verification
- Property-based testing with random inputs

### Entropy Pooling Tests ♡(ᐢ ᴗ ᐢ )
- XOR correctness: `A ⊕ B = expected`
- Mathematical properties: 
  - `A ⊕ A = 0` (identity)
  - `A ⊕ 0 = A` (zero property)
  - `A ⊕ B = B ⊕ A` (commutative)
- Multiple source pooling

### Multi-Stage Extraction Tests ⋆⁺₊⋆
- Deterministic extraction (same input = same output)
- Output length validation (SHA-512 = 64 bytes)
- Small input expansion
- Bit distribution balance (45-55% ones)

### All 16 NIST Tests ✧･ﾟ: *✧･ﾟ:*
Each test validated with:
- Balanced sequences (should pass) (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
- Biased sequences (should fail) (｡•́‿•̀｡)
- Random data (realistic scenarios) ໒꒰ྀི´ ˘ ` ꒱ྀīა

| Test | What It Checks |
|------|----------------|
| Monobit | Equal 1s and 0s |
| Runs | Bit flip frequency |
| Longest Run | Streak patterns |
| Spectral | Hidden periodicities (FFT) |
| Block Frequency | Local balance |
| Cumulative Sums | Random walk behavior |
| Chi-Square | Basic distribution |
| Serial Correlation | Bit independence |
| Poker | Pattern distribution |
| Gap | Spacing analysis |
| Autocorrelation | Distance correlation |
| Shannon Entropy | Information content |
| Matrix Rank | Linear independence |
| Linear Complexity | Unpredictability |
| Serial (m=3) | Advanced pattern test |
| Maurer's Universal | Compressibility |

### Cryptographic Quality Tests ✨
- **Avalanche effect**: Single bit change → ~50% output change
- **Key uniqueness**: 100 keys generated, 100 unique
- **Hamming distance**: Sequential keys differ by ~50%
- **Bit balance**: All keys have 45-55% ones

### Integration Tests (っ◔◡◔)っ
- Full pipeline: visual entropy → debias → pool → extract
- CSV persistence roundtrip
- Length invariants maintained
- Low correlation between sequential keys

---

## Test Statistics ⋆｡˚ ☁︎ ˚｡⋆

```
Total Tests: 100+
Coverage Areas:
- Unit tests: Von Neumann, pooling, extraction ✧
- Statistical tests: All 16 NIST SP 800-22 ☆
- Cryptographic: Avalanche, uniqueness, Hamming ♡
- Integration: End-to-end pipelines (˶ᵔ ᵕ ᵔ˶)
- Property-based: Random input validation ✮
- Performance: Benchmarks included ⸜(｡˃ ᵕ ˂ )⸝
```

---


---

## Why These Tests Matter ໒꒰ྀི ˶• ˕ •˶ ꒱ྀīა

### For Researchers ✧
- Validates your entropy extraction methods
- Proves statistical properties mathematically
- Demonstrates NIST compliance
- Provides reproducible results

### For Developers ⸜(｡˃ ᵕ ˂ )⸝
- Catches bugs in crypto implementations
- Verifies key generation quality
- Prevents weak keys from being used
- Ensures avalanche effect works

### For Security ☆ﾟ.*･｡ﾟ
- Tests that keys are truly random
- Validates no hidden patterns exist
- Checks cryptographic properties hold
- Ensures keys can't be predicted

---

## Understanding Test Results ･ﾟ( ﾉд`ﾟ)

```python
# Good test output ✧
PASSED test_monobit_balanced_passes
PASSED test_avalanche_effect_verified  
PASSED test_key_uniqueness_statistical

# If you see failures (╥﹏╥)
FAILED test_monobit_balanced_passes
→ Your entropy source is biased!

FAILED test_avalanche_effect_verified
→ Hash function not working properly!

FAILED test_key_uniqueness_statistical
→ Keys are repeating - major problem!
```

---

## Running Specific Test Suites ♪(´▽｀)

```bash
# Just Von Neumann tests
pytest test_quantum_key_generator.py::TestVonNeumannDebiasing -v

# Just NIST statistical tests
pytest test_quantum_key_generator.py::TestNISTStatisticalTests -v

# Just cryptographic quality
pytest test_quantum_key_generator.py::TestCryptographicQuality -v

# Integration tests only
pytest test_quantum_key_generator.py::TestIntegration -v

# With detailed output
pytest test_quantum_key_generator.py -vv

# Stop on first failure
pytest test_quantum_key_generator.py -x
```

---

## For Research Papers (˶ᵔ ᵕ ᵔ˶)

This test suite provides:
- **Reproducible results**: All tests are deterministic ✧
- **Statistical validation**: NIST SP 800-22 compliance ☆
- **Cryptographic proofs**: Avalanche, uniqueness, Hamming distance ♡
- **Performance metrics**: Benchmark data included ⋆
- **Comprehensive coverage**: 100+ tests across all components ✮

Perfect for documenting your key generation system in academic papers!

---

## Dependencies ･ﾟ✧

```
pytest >= 7.0.0
numpy >= 1.20.0
pandas >= 1.3.0
scipy >= 1.7.0
hypothesis >= 6.0.0
pytest-benchmark >= 3.4.0
```


## License & Usage ⸜(｡˃ ᵕ ˂ )⸝

Free to use for:
- Research projects ✧
- Learning about cryptography ☆
- Testing your own key generators ♡
- Academic papers (please cite!) (ˆ⌣ˆ)

---

Made with ♡ for secure randomness testing ໒꒰ྀི ˶• ˕ •˶ ꒱ྀīა

*Because predictable keys are scary* (ﾉД`)･ﾟ･｡
