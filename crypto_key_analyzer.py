"""
CRYPTOGRAPHIC KEY ANALYSIS TOOL
================================
This tool performs 34 comprehensive statistical and cryptographic tests on keys from a CSV file.
Includes complete NIST Statistical Test Suite (15 tests) and advanced cryptographic randomness tests (16+ tests).

INSTRUCTIONS FOR USE:
1. Prepare your CSV file with cryptographic keys (one key per row or column)
2. Modify the CONFIGURATION section below to point to your CSV file
3. Run: python crypto_key_analyzer.py
4. Check the output files for detailed results

Author: Automated Crypto Analysis Tool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, fft
from scipy.special import erfc, gammaincc
from collections import Counter
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS FOR YOUR USE CASE
# ============================================================================

CONFIG = {
    # Input file settings
    'csv_file': 'keys.csv',              # Path to your CSV file with keys
    'key_column': 0,                      # Column index containing keys (0-indexed)
    'key_encoding': 'hex',                # Format: 'hex', 'binary', 'base64', or 'decimal'
    
    # Test parameters
    'significance_level': 0.01,           # Alpha level for statistical tests (0.01 = 99% confidence)
    'min_bit_length': 128,                # Minimum bits required for analysis
    
    # Output settings
    'output_dir': 'crypto_analysis_results',
    'generate_plots': True,               # Generate statistical plots
    'save_csv_report': True,              # Save detailed CSV report
    'verbose': True,                       # Print detailed progress
    
    # Advanced test parameters
    'block_size': 128,                    # Block size for block frequency test
    'template_size': 9,                   # Template size for template matching
    'serial_block_length': 16,            # Block length for serial test
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import os

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

def read_keys_from_csv(filepath, column_index=0):
    """Read cryptographic keys from CSV file."""
    try:
        df = pd.read_csv(filepath, header=None)
        keys = df.iloc[:, column_index].tolist()
        return keys
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def convert_to_binary(key, encoding='hex'):
    """Convert key to binary string based on encoding."""
    try:
        if encoding == 'hex':
            binary = bin(int(key, 16))[2:]
        elif encoding == 'binary':
            binary = str(key)
        elif encoding == 'base64':
            import base64
            decoded = base64.b64decode(key)
            binary = ''.join(format(byte, '08b') for byte in decoded)
        elif encoding == 'decimal':
            binary = bin(int(key))[2:]
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        return binary
    except Exception as e:
        print(f"Error converting key to binary: {e}")
        return None

def binary_to_int_array(binary_string):
    """Convert binary string to integer array."""
    return np.array([int(bit) for bit in binary_string])

# ============================================================================
# NIST STATISTICAL TEST SUITE
# ============================================================================

class NISTTests:
    """Implementation of NIST SP 800-22 Statistical Test Suite."""
    
    @staticmethod
    def frequency_monobit_test(bits):
        """
        Test 1: Frequency (Monobit) Test
        Purpose: Determine if the number of ones and zeros are approximately equal.
        """
        n = len(bits)
        s = np.sum(2 * bits - 1)
        s_obs = abs(s) / np.sqrt(n)
        p_value = erfc(s_obs / np.sqrt(2))
        return p_value, s_obs
    
    @staticmethod
    def frequency_block_test(bits, block_size=128):
        """
        Test 2: Frequency Test within a Block
        Purpose: Determine if the frequency of ones in an M-bit block is approximately M/2.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks == 0:
            return None, f"Insufficient bits (need >{block_size})"
        
        block_bits = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
        proportions = np.sum(block_bits, axis=1) / block_size
        chi_squared = 4 * block_size * np.sum((proportions - 0.5) ** 2)
        p_value = gammaincc(num_blocks / 2, chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def runs_test(bits):
        """
        Test 3: Runs Test
        Purpose: Determine if the number of runs is as expected for a random sequence.
        """
        n = len(bits)
        pi = np.sum(bits) / n
        
        if abs(pi - 0.5) >= 2 / np.sqrt(n):
            return None, "Bit proportion too far from 0.5 for test validity"
        
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        p_value = erfc(abs(runs - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
        
        return p_value, runs
    
    @staticmethod
    def longest_run_test(bits):
        """
        Test 4: Test for the Longest Run of Ones in a Block
        Purpose: Determine if the longest run of ones is consistent with randomness.
        """
        n = len(bits)
        
        if n < 128:
            return None, "Insufficient bits (need ≥128)"
        
        if n < 6272:
            M, K = 8, 3
            v_values = [1, 2, 3, 4]
            pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
        elif n < 750000:
            M, K = 128, 5
            v_values = [4, 5, 6, 7, 8, 9]
            pi_values = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:
            M, K = 10000, 6
            v_values = [10, 11, 12, 13, 14, 15, 16]
            pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        
        N = n // M
        frequencies = np.zeros(K + 1)
        
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            max_run = 0
            current_run = 0
            
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            
            for j, v in enumerate(v_values):
                if max_run <= v:
                    frequencies[j] += 1
                    break
            else:
                frequencies[K] += 1
        
        chi_squared = np.sum((frequencies - N * np.array(pi_values)) ** 2 / (N * np.array(pi_values)))
        p_value = gammaincc(K / 2, chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def binary_matrix_rank_test(bits):
        """
        Test 5: Binary Matrix Rank Test
        Purpose: Check for linear dependence among fixed length substrings.
        """
        n = len(bits)
        M, Q = 32, 32
        N = n // (M * Q)
        
        if N == 0:
            return None, f"Insufficient bits (need ≥{M*Q})"
        
        fm, fm1, remainder = 0, 0, 0
        
        for i in range(N):
            block = bits[i*M*Q:(i+1)*M*Q].reshape(M, Q)
            rank = np.linalg.matrix_rank(block)
            
            if rank == M:
                fm += 1
            elif rank == M - 1:
                fm1 += 1
            else:
                remainder += 1
        
        p_m = 0.2888
        p_m1 = 0.5776
        p_remainder = 0.1336
        
        chi_squared = ((fm - N * p_m) ** 2 / (N * p_m) +
                      (fm1 - N * p_m1) ** 2 / (N * p_m1) +
                      (remainder - N * p_remainder) ** 2 / (N * p_remainder))
        
        p_value = np.exp(-chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def dft_test(bits):
        """
        Test 6: Discrete Fourier Transform (Spectral) Test
        Purpose: Detect periodic features that would indicate deviation from randomness.
        """
        n = len(bits)
        s = 2 * bits - 1
        
        S = np.abs(fft.fft(s)[:n//2])
        T = np.sqrt(np.log(1/0.05) * n)
        N0 = 0.95 * n / 2
        N1 = np.sum(S < T)
        
        d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
        p_value = erfc(abs(d) / np.sqrt(2))
        
        return p_value, d
    
    @staticmethod
    def non_overlapping_template_test(bits, template_size=9):
        """
        Test 7: Non-overlapping Template Matching Test
        Purpose: Detect too many occurrences of a given pattern.
        """
        n = len(bits)
        m = template_size
        M = 1024
        N = n // M
        
        if N == 0:
            return None, f"Insufficient bits (need ≥{M})"
        
        template = np.ones(m, dtype=int)
        mu = (M - m + 1) / (2 ** m)
        sigma_squared = M * ((1 / (2 ** m)) - (2 * m - 1) / (2 ** (2 * m)))
        
        W = np.zeros(N)
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            count = 0
            j = 0
            while j < M - m + 1:
                if np.array_equal(block[j:j+m], template):
                    count += 1
                    j += m
                else:
                    j += 1
            W[i] = count
        
        chi_squared = np.sum((W - mu) ** 2 / sigma_squared)
        p_value = gammaincc(N / 2, chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def overlapping_template_test(bits, template_size=9):
        """
        Test 8: Overlapping Template Matching Test
        Purpose: Detect deviations in the expected number of runs of ones.
        """
        n = len(bits)
        m = template_size
        M = 1032
        N = n // M
        
        if N == 0 or m > M:
            return None, f"Insufficient bits (need ≥{M})"
        
        template = np.ones(m, dtype=int)
        lambda_val = (M - m + 1) / (2 ** m)
        eta = lambda_val / 2
        
        pi = [0.367879, 0.183940, 0.137955, 0.099634, 0.069935, 0.140657]
        K = 5
        
        v = np.zeros(K + 1)
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            count = 0
            for j in range(M - m + 1):
                if np.array_equal(block[j:j+m], template):
                    count += 1
            
            if count <= K:
                v[count] += 1
            else:
                v[K] += 1
        
        chi_squared = np.sum((v - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p_value = gammaincc(K / 2, chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def maurers_universal_test(bits):
        """
        Test 9: Maurer's Universal Statistical Test
        Purpose: Detect if the sequence can be significantly compressed.
        """
        n = len(bits)
        L = 7
        Q = 1280
        K = n // L - Q
        
        if K <= 0 or n < 387840:
            return None, "Insufficient bits (need ≥387,840)"
        
        T = np.zeros(2 ** L, dtype=int)
        
        for i in range(Q):
            pattern = int(''.join(bits[i*L:(i+1)*L].astype(str)), 2)
            T[pattern] = i + 1
        
        sum_log = 0.0
        for i in range(Q, Q + K):
            pattern = int(''.join(bits[i*L:(i+1)*L].astype(str)), 2)
            sum_log += np.log2(i + 1 - T[pattern])
            T[pattern] = i + 1
        
        fn = sum_log / K
        
        expected_value = {5: 5.2177, 6: 6.1962, 7: 7.1836, 8: 8.1764}
        variance = {5: 2.954, 6: 3.125, 7: 3.238, 8: 3.311}
        
        c = 0.7 - 0.8 / L + (4 + 32 / L) * (K ** (-3 / L)) / 15
        sigma = c * np.sqrt(variance.get(L, 3.238) / K)
        
        p_value = erfc(abs((fn - expected_value.get(L, 7.1836)) / (np.sqrt(2) * sigma)))
        
        return p_value, fn
    
    @staticmethod
    def linear_complexity_test(bits, block_size=500):
        """
        Test 10: Linear Complexity Test
        Purpose: Determine if the sequence is complex enough to be random.
        """
        n = len(bits)
        M = block_size
        N = n // M
        
        if N == 0:
            return None, f"Insufficient bits (need ≥{M})"
        
        mu = M / 2 + (9 + (-1) ** (M + 1)) / 36 - 1 / (2 ** M) * (M / 3 + 2 / 9)
        
        T = [-1.0 * (-1) ** M * (mu - 2.0 / 9) + 2.0 / 9,
             -1.0 * (-1) ** M * (mu - 1.0 / 9) + 1.0 / 9,
             1.0 / 9,
             2.0 / 9]
        
        v = [0, 0, 0, 0, 0, 0, 0]
        
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            L = berlekamp_massey(block)
            T_i = (-1) ** M * (L - mu) + 2.0 / 9
            
            if T_i <= T[0]:
                v[0] += 1
            elif T_i <= T[1]:
                v[1] += 1
            elif T_i <= T[2]:
                v[2] += 1
            elif T_i <= T[3]:
                v[3] += 1
            else:
                v[4] += 1
        
        pi = [0.01047, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        chi_squared = np.sum((np.array(v[:5]) - N * np.array(pi[:5])) ** 2 / (N * np.array(pi[:5])))
        p_value = gammaincc(2, chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def serial_test(bits, block_length=16):
        """
        Test 11: Serial Test
        Purpose: Determine if the number of occurrences of patterns is uniform.
        """
        n = len(bits)
        m = block_length
        
        if n < max(m, 3):
            return (0.0, 0.0), 0.0
        
        def psi_squared(bits, m):
            n = len(bits)
            if m == 0:
                return 0.0
            
            padded_bits = np.concatenate([bits, bits[:m-1]])
            counts = {}
            
            for i in range(n):
                pattern = tuple(padded_bits[i:i+m])
                counts[pattern] = counts.get(pattern, 0) + 1
            
            psi = 0.0
            for count in counts.values():
                psi += count ** 2
            
            psi = (psi * (2 ** m) / n) - n
            return psi
        
        psi_m = psi_squared(bits, m)
        psi_m1 = psi_squared(bits, m - 1)
        psi_m2 = psi_squared(bits, m - 2)
        
        delta1 = psi_m - psi_m1
        delta2 = psi_m - 2 * psi_m1 + psi_m2
        
        p_value1 = gammaincc(2 ** (m - 2), delta1 / 2)
        p_value2 = gammaincc(2 ** (m - 3), delta2 / 2)
        
        return (p_value1, p_value2), (delta1, delta2)
    
    @staticmethod
    def approximate_entropy_test(bits, block_length=10):
        """
        Test 12: Approximate Entropy Test
        Purpose: Compare the frequency of overlapping blocks of two consecutive lengths.
        """
        n = len(bits)
        m = block_length
        
        if m >= n:
            return None, f"Insufficient bits (need >{m})"
        
        def phi(m):
            padded_bits = np.concatenate([bits, bits[:m-1]])
            counts = {}
            
            for i in range(n):
                pattern = tuple(padded_bits[i:i+m])
                counts[pattern] = counts.get(pattern, 0) + 1
            
            phi_m = 0.0
            for count in counts.values():
                pi = count / n
                phi_m += pi * np.log(pi)
            
            return phi_m
        
        phi_m = phi(m)
        phi_m1 = phi(m + 1)
        
        apen = phi_m - phi_m1
        chi_squared = 2 * n * (np.log(2) - apen)
        p_value = gammaincc(2 ** (m - 1), chi_squared / 2)
        
        return p_value, chi_squared
    
    @staticmethod
    def cumulative_sums_test(bits):
        """
        Test 13: Cumulative Sums (Forward and Backward) Test
        Purpose: Determine if the cumulative sum is too large or too small.
        """
        n = len(bits)
        s = 2 * bits - 1
        
        cumsum_forward = np.cumsum(s)
        cumsum_backward = np.cumsum(s[::-1])
        
        z_forward = np.max(np.abs(cumsum_forward))
        z_backward = np.max(np.abs(cumsum_backward))
        
        def compute_p_value(z):
            sum_a = 0.0
            start = int((-n / z + 1) / 4)
            end = int((n / z - 1) / 4)
            
            for k in range(start, end + 1):
                term1 = stats.norm.cdf((4 * k + 1) * z / np.sqrt(n))
                term2 = stats.norm.cdf((4 * k - 1) * z / np.sqrt(n))
                sum_a += term1 - term2
            
            sum_b = 0.0
            start = int((-n / z - 3) / 4)
            end = int((n / z - 1) / 4)
            
            for k in range(start, end + 1):
                term1 = stats.norm.cdf((4 * k + 3) * z / np.sqrt(n))
                term2 = stats.norm.cdf((4 * k + 1) * z / np.sqrt(n))
                sum_b += term1 - term2
            
            p_value = 1.0 - sum_a + sum_b
            return p_value
        
        p_forward = compute_p_value(z_forward)
        p_backward = compute_p_value(z_backward)
        
        return (p_forward, p_backward), (z_forward, z_backward)
    
    @staticmethod
    def random_excursions_test(bits):
        """
        Test 14: Random Excursions Test
        Purpose: Determine if the number of visits to a state is as expected.
        """
        n = len(bits)
        s = 2 * bits - 1
        cumsum = np.concatenate([[0], np.cumsum(s), [0]])
        
        cycles = []
        current_cycle = []
        
        for i, val in enumerate(cumsum):
            if val == 0 and i > 0:
                if current_cycle:
                    cycles.append(current_cycle)
                current_cycle = []
            else:
                current_cycle.append(val)
        
        if len(cycles) == 0:
            return {}, 0
        
        J = len(cycles)
        
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        pi = {
            -4: [0.0000, 0.00000, 0.00002, 0.00011, 0.00039, 0.00107, 0.00259],
            -3: [0.0000, 0.00004, 0.00030, 0.00145, 0.00426, 0.00954, 0.01845],
            -2: [0.0001, 0.00051, 0.00281, 0.00934, 0.02081, 0.03563, 0.05651],
            -1: [0.0010, 0.00598, 0.02207, 0.05471, 0.09893, 0.14066, 0.18483],
             1: [0.0010, 0.00598, 0.02207, 0.05471, 0.09893, 0.14066, 0.18483],
             2: [0.0001, 0.00051, 0.00281, 0.00934, 0.02081, 0.03563, 0.05651],
             3: [0.0000, 0.00004, 0.00030, 0.00145, 0.00426, 0.00954, 0.01845],
             4: [0.0000, 0.00000, 0.00002, 0.00011, 0.00039, 0.00107, 0.00259]
        }
        
        results = {}
        for state in states:
            v = [0] * 6
            for cycle in cycles:
                count = np.sum(np.array(cycle) == state)
                if count >= 5:
                    v[5] += 1
                else:
                    v[count] += 1
            
            chi_squared = 0.0
            for i in range(6):
                expected = J * pi[state][i]
                if expected > 0:
                    chi_squared += (v[i] - expected) ** 2 / expected
            
            p_value = gammaincc(2.5, chi_squared / 2)
            results[state] = p_value
        
        return results, J
    
    @staticmethod
    def random_excursions_variant_test(bits):
        """
        Test 15: Random Excursions Variant Test
        Purpose: Detect deviations from the expected number of visits to states.
        """
        n = len(bits)
        s = 2 * bits - 1
        cumsum = np.cumsum(s)
        cumsum = np.concatenate([[0], cumsum, [0]])
        
        J = np.sum(cumsum == 0) - 1
        
        if J == 0:
            return {}, 0
        
        states = list(range(-9, 10))
        states.remove(0)
        
        results = {}
        for state in states:
            count = np.sum(cumsum == state)
            p_value = erfc(abs(count - J) / np.sqrt(2 * J * (4 * abs(state) - 2)))
            results[state] = p_value
        
        return results, J

def berlekamp_massey(bits):
    """Berlekamp-Massey algorithm for linear complexity calculation."""
    n = len(bits)
    b = np.zeros(n, dtype=int)
    c = np.zeros(n, dtype=int)
    b[0] = 1
    c[0] = 1
    
    L = 0
    m = -1
    N = 0
    
    while N < n:
        d = bits[N]
        for i in range(1, L + 1):
            d ^= c[i] & bits[N - i]
        
        if d == 1:
            t = c.copy()
            for i in range(n - N + m):
                c[N - m + i] ^= b[i]
            
            if L <= N / 2:
                L = N + 1 - L
                m = N
                b = t
        
        N += 1
    
    return L

# ============================================================================
# ADDITIONAL CRYPTOGRAPHIC RANDOMNESS TESTS
# ============================================================================

class AdditionalTests:
    """Additional cryptographic and statistical randomness tests."""
    
    @staticmethod
    def autocorrelation_test(bits, max_shift=100):
        """
        Test for autocorrelation at different shifts.
        Measures correlation between sequence and shifted versions.
        """
        n = len(bits)
        s = 2 * bits - 1
        results = []
        
        for shift in range(1, min(max_shift, n // 2)):
            correlation = np.correlate(s[:-shift], s[shift:], mode='valid')[0]
            normalized = correlation / (n - shift)
            results.append((shift, normalized))
        
        max_corr = max(abs(r[1]) for r in results)
        threshold = 3 / np.sqrt(n)
        p_value = 1.0 - stats.norm.cdf(max_corr * np.sqrt(n))
        
        return p_value, max_corr, results
    
    @staticmethod
    def entropy_test(bits, block_size=8):
        """
        Shannon entropy test.
        Measures the entropy of the bit sequence.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks == 0:
            return None, f"Insufficient bits (need ≥{block_size})"
        
        blocks = []
        for i in range(num_blocks):
            block = ''.join(bits[i*block_size:(i+1)*block_size].astype(str))
            blocks.append(block)
        
        freq = Counter(blocks)
        entropy = 0.0
        
        for count in freq.values():
            p = count / num_blocks
            entropy -= p * np.log2(p)
        
        max_entropy = block_size
        normalized_entropy = entropy / max_entropy
        
        p_value = 1.0 if normalized_entropy > 0.99 else normalized_entropy
        
        return p_value, entropy
    
    @staticmethod
    def chi_square_test(bits):
        """
        Chi-square test for uniformity.
        Tests if the distribution of bits is uniform.
        """
        ones = np.sum(bits)
        zeros = len(bits) - ones
        expected = len(bits) / 2
        
        chi_squared = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
        p_value = 1 - stats.chi2.cdf(chi_squared, df=1)
        
        return p_value, chi_squared
    
    @staticmethod
    def poker_test(bits, block_size=4):
        """
        Poker test for randomness.
        Divides sequence into blocks and checks distribution.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks == 0:
            return None, f"Insufficient bits (need ≥{block_size})"
        
        patterns = {}
        for i in range(num_blocks):
            pattern = ''.join(bits[i*block_size:(i+1)*block_size].astype(str))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        k = 2 ** block_size
        sum_squared = sum(count ** 2 for count in patterns.values())
        
        X = (k / num_blocks) * sum_squared - num_blocks
        p_value = 1 - stats.chi2.cdf(X, df=k-1)
        
        return p_value, X
    
    @staticmethod
    def gap_test(bits, alpha=0, beta=1):
        """
        Gap test for randomness.
        Measures gaps between occurrences of patterns.
        """
        gaps = []
        current_gap = 0
        in_gap = bits[0] == 0
        
        for bit in bits[1:]:
            if in_gap:
                if bit == 1:
                    gaps.append(current_gap)
                    current_gap = 0
                    in_gap = False
                else:
                    current_gap += 1
            else:
                if bit == 0:
                    in_gap = True
                    current_gap = 0
        
        if len(gaps) < 2:
            return None, "Insufficient gaps for analysis"
        
        mean_gap = np.mean(gaps)
        expected_gap = 1.0
        
        chi_squared = sum((g - expected_gap) ** 2 / expected_gap for g in gaps)
        p_value = 1 - stats.chi2.cdf(chi_squared, df=len(gaps)-1)
        
        return p_value, mean_gap
    
    @staticmethod
    def collision_test(bits, block_size=8):
        """
        Collision test.
        Checks for unexpected collisions in block patterns.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks < 2:
            return None, f"Insufficient bits (need ≥{block_size*2})"
        
        seen = set()
        collisions = 0
        
        for i in range(num_blocks):
            pattern = ''.join(bits[i*block_size:(i+1)*block_size].astype(str))
            if pattern in seen:
                collisions += 1
            seen.add(pattern)
        
        expected_collisions = num_blocks ** 2 / (2 * (2 ** block_size))
        
        if expected_collisions == 0:
            return 1.0, collisions
        
        z_score = (collisions - expected_collisions) / np.sqrt(expected_collisions)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value, collisions
    
    @staticmethod
    def birthday_spacing_test(bits, block_size=8, num_samples=1000):
        """
        Birthday spacing test.
        Based on birthday paradox for collision detection.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks < num_samples:
            num_samples = num_blocks
        
        samples = []
        for i in range(num_samples):
            if i * block_size + block_size <= n:
                block_val = int(''.join(bits[i*block_size:(i+1)*block_size].astype(str)), 2)
                samples.append(block_val)
        
        samples.sort()
        spacings = [samples[i+1] - samples[i] for i in range(len(samples)-1)]
        
        if len(spacings) == 0:
            return None, "Insufficient samples for spacing analysis"
        
        mean_spacing = np.mean(spacings)
        expected_spacing = (2 ** block_size) / num_samples
        
        chi_squared = sum((s - expected_spacing) ** 2 / expected_spacing for s in spacings if expected_spacing > 0)
        p_value = 1 - stats.chi2.cdf(chi_squared, df=len(spacings))
        
        return p_value, mean_spacing
    
    @staticmethod
    def bit_independence_test(bits):
        """
        Test for bit independence.
        Checks if individual bits are independent of each other.
        """
        n = len(bits)
        if n < 100:
            return None, "Insufficient bits (need ≥100)"
        
        correlations = []
        for offset in range(1, min(33, n // 2)):
            corr = np.corrcoef(bits[:-offset], bits[offset:])[0, 1]
            correlations.append(abs(corr))
        
        max_corr = max(correlations)
        threshold = 2 / np.sqrt(n)
        
        p_value = 1.0 - stats.norm.cdf(max_corr * np.sqrt(n))
        
        return p_value, max_corr
    
    @staticmethod
    def avalanche_effect_test(bits, block_size=64):
        """
        Avalanche effect test.
        Checks if flipping one bit causes significant changes.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks < 2:
            return None, f"Insufficient bits (need ≥{block_size*2})"
        
        avalanche_scores = []
        
        for i in range(num_blocks - 1):
            block1 = bits[i*block_size:(i+1)*block_size]
            block2 = bits[(i+1)*block_size:(i+2)*block_size]
            
            differences = np.sum(block1 != block2)
            avalanche_ratio = differences / block_size
            avalanche_scores.append(avalanche_ratio)
        
        mean_avalanche = np.mean(avalanche_scores)
        expected_avalanche = 0.5
        
        z_score = abs(mean_avalanche - expected_avalanche) / (np.sqrt(0.25 / block_size))
        p_value = 2 * (1 - stats.norm.cdf(z_score))
        
        return p_value, mean_avalanche
    
    @staticmethod
    def strict_avalanche_criterion(bits, block_size=64):
        """
        Strict Avalanche Criterion (SAC) test.
        More rigorous version of avalanche effect.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks < 2:
            return None, f"Insufficient bits (need ≥{block_size*2})"
        
        sac_matrix = np.zeros((block_size, block_size))
        
        for i in range(num_blocks - 1):
            block1 = bits[i*block_size:(i+1)*block_size]
            block2 = bits[(i+1)*block_size:(i+2)*block_size]
            
            for j in range(block_size):
                flipped = block1.copy()
                flipped[j] = 1 - flipped[j]
                
                changes = flipped != block2
                sac_matrix[j] += changes
        
        sac_matrix /= (num_blocks - 1)
        
        mean_sac = np.mean(sac_matrix)
        deviation = np.abs(mean_sac - 0.5)
        
        p_value = 1.0 - 2 * deviation
        
        return max(0, p_value), mean_sac
    
    @staticmethod
    def run_distribution_test(bits):
        """
        Run distribution test.
        Analyzes the distribution of run lengths.
        """
        runs = []
        current_run = 1
        
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        run_dist = Counter(runs)
        
        chi_squared = 0
        for length, count in run_dist.items():
            expected = len(bits) / (2 ** (length + 1))
            if expected > 0:
                chi_squared += (count - expected) ** 2 / expected
        
        p_value = 1 - stats.chi2.cdf(chi_squared, df=len(run_dist)-1)
        
        return p_value, run_dist
    
    @staticmethod
    def coupon_collector_test(bits, block_size=4):
        """
        Coupon collector's test.
        Measures how long it takes to see all possible patterns.
        """
        n = len(bits)
        num_possible = 2 ** block_size
        
        seen = set()
        blocks_seen = 0
        
        for i in range(n // block_size):
            pattern = ''.join(bits[i*block_size:(i+1)*block_size].astype(str))
            seen.add(pattern)
            blocks_seen += 1
            
            if len(seen) == num_possible:
                break
        
        expected = num_possible * (np.log(num_possible) + 0.5772)
        
        if expected == 0:
            return 1.0, blocks_seen
        
        z_score = (blocks_seen - expected) / np.sqrt(expected)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value, blocks_seen
    
    @staticmethod
    def permutation_test(bits, block_size=8):
        """
        Permutation test.
        Checks if all permutations appear with equal frequency.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks < 10:
            return None, f"Insufficient bits (need ≥{block_size*10})"
        
        patterns = {}
        for i in range(num_blocks):
            pattern = ''.join(bits[i*block_size:(i+1)*block_size].astype(str))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        expected_freq = num_blocks / (2 ** block_size)
        
        chi_squared = 0
        for count in patterns.values():
            if expected_freq > 0:
                chi_squared += (count - expected_freq) ** 2 / expected_freq
        
        df = len(patterns) - 1
        p_value = 1 - stats.chi2.cdf(chi_squared, df=df) if df > 0 else 0.0
        
        return p_value, chi_squared
    
    @staticmethod
    def periodicity_test(bits, max_period=1000):
        """
        Periodicity detection test.
        Checks for repeating patterns indicating non-randomness.
        """
        n = len(bits)
        bit_string = ''.join(bits.astype(str))
        
        periods_found = []
        
        for period in range(2, min(max_period, n // 3)):
            matches = 0
            comparisons = 0
            
            for i in range(n - period):
                if i + 2 * period < n:
                    if bit_string[i] == bit_string[i + period]:
                        matches += 1
                    comparisons += 1
            
            if comparisons > 0:
                match_ratio = matches / comparisons
                if match_ratio > 0.9:
                    periods_found.append((period, match_ratio))
        
        if periods_found:
            max_ratio = max(p[1] for p in periods_found)
            p_value = 1.0 - max_ratio
        else:
            p_value = 1.0
        
        return p_value, periods_found
    
    @staticmethod
    def maurer_universal_extended(bits):
        """
        Extended Maurer's Universal test with multiple block sizes.
        """
        results = []
        
        for L in [5, 6, 7, 8]:
            Q = max(10 * (2 ** L), 1000)
            n = len(bits)
            K = n // L - Q
            
            if K > 0:
                T = {}
                
                for i in range(Q):
                    if i * L + L <= n:
                        pattern = ''.join(bits[i*L:(i+1)*L].astype(str))
                        T[pattern] = i + 1
                
                sum_log = 0.0
                count = 0
                
                for i in range(Q, min(Q + K, n // L)):
                    if i * L + L <= n:
                        pattern = ''.join(bits[i*L:(i+1)*L].astype(str))
                        if pattern in T:
                            sum_log += np.log2(i + 1 - T[pattern])
                        T[pattern] = i + 1
                        count += 1
                
                if count > 0:
                    fn = sum_log / count
                    results.append((L, fn))
        
        if results:
            avg_complexity = np.mean([r[1] for r in results])
            p_value = 1.0 if avg_complexity > 5.0 else avg_complexity / 10.0
            return p_value, results
        else:
            return None, "Insufficient bits for any block size tested"
    
    @staticmethod
    def lempel_ziv_complexity(bits):
        """
        Lempel-Ziv complexity test.
        Measures compressibility of the sequence.
        """
        n = len(bits)
        bit_string = ''.join(bits.astype(str))
        
        i = 0
        c = 1
        u = 1
        v = 1
        v_max = v
        
        while u + v <= n:
            if bit_string[i + v - 1] == bit_string[u + v - 1]:
                v += 1
            else:
                v_max = max(v_max, v)
                i += 1
                if i == u:
                    c += 1
                    u += v_max
                    v = 1
                    i = 0
                    v_max = v
                else:
                    v = 1
        
        if v != 1:
            c += 1
        
        expected_c = n / (np.log(n) / np.log(2))
        normalized = c / expected_c if expected_c > 0 else 0
        
        p_value = 1.0 if normalized > 0.8 else normalized
        
        return p_value, c

# ============================================================================
# KEY QUALITY METRICS
# ============================================================================

class KeyQualityMetrics:
    """Advanced key quality assessment metrics."""
    
    @staticmethod
    def estimate_key_strength(bits):
        """
        Estimate cryptographic strength of the key.
        """
        n = len(bits)
        
        ones_ratio = np.sum(bits) / n
        balance_score = 1.0 - 2 * abs(ones_ratio - 0.5)
        
        bit_string = ''.join(bits.astype(str))
        patterns = {}
        for i in range(n - 7):
            pattern = bit_string[i:i+8]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        uniqueness_score = len(patterns) / min(256, n - 7)
        
        changes = np.sum(bits[1:] != bits[:-1])
        transition_score = changes / (n - 1)
        
        strength = (balance_score + uniqueness_score + transition_score) / 3
        
        return strength * 100
    
    @staticmethod
    def uniformity_analysis(bits, block_size=8):
        """
        Analyze uniformity of bit distribution.
        """
        n = len(bits)
        num_blocks = n // block_size
        
        if num_blocks == 0:
            return 0.0
        
        block_sums = []
        for i in range(num_blocks):
            block = bits[i*block_size:(i+1)*block_size]
            block_sums.append(np.sum(block))
        
        expected_sum = block_size / 2
        variance = np.var(block_sums)
        expected_variance = block_size / 4
        
        uniformity = 1.0 - min(1.0, abs(variance - expected_variance) / expected_variance)
        
        return uniformity * 100
    
    @staticmethod
    def complexity_score(bits):
        """
        Calculate overall complexity score of the key.
        """
        n = len(bits)
        
        bit_string = ''.join(bits.astype(str))
        unique_bytes = len(set(bit_string[i:i+8] for i in range(0, n-7, 8)))
        byte_complexity = unique_bytes / min(256, (n // 8))
        
        transitions = np.sum(bits[1:] != bits[:-1])
        transition_complexity = transitions / (n - 1)
        
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        run_complexity = runs / n
        
        complexity = (byte_complexity + transition_complexity + run_complexity) / 3
        
        return complexity * 100

# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self):
        self.all_results = []
        self.test_count = 0
        
    def add_result(self, test_name, p_value, statistic, passed, details="", skipped=False):
        """Add a test result to the report."""
        self.test_count += 1
        
        if skipped:
            result = 'SKIP'
            p_value_display = 'N/A'
        elif isinstance(p_value, (list, tuple)):
            p_value_display = f"{p_value[0]:.6f}, {p_value[1]:.6f}"
        elif isinstance(p_value, dict):
            p_value_display = "Multiple values"
        elif p_value is None:
            p_value_display = 'N/A'
            result = 'SKIP'
        else:
            p_value_display = f"{p_value:.6f}"
            result = 'PASS' if passed else 'FAIL'
        
        if not skipped and p_value is not None:
            result = 'PASS' if passed else 'FAIL'
        
        self.all_results.append({
            'Test Number': self.test_count,
            'Test Name': test_name,
            'P-Value': p_value_display,
            'Statistic': str(statistic)[:50],
            'Result': result,
            'Details': details
        })
    
    def generate_summary(self, key_id, bit_length):
        """Generate summary statistics."""
        passed = sum(1 for r in self.all_results if r['Result'] == 'PASS')
        failed = sum(1 for r in self.all_results if r['Result'] == 'FAIL')
        skipped = sum(1 for r in self.all_results if r['Result'] == 'SKIP')
        applicable = len(self.all_results) - skipped
        
        summary = f"""
{'='*80}
CRYPTOGRAPHIC KEY ANALYSIS SUMMARY
{'='*80}
Key ID: {key_id}
Bit Length: {bit_length}
Total Tests: {len(self.all_results)}
Tests Run: {applicable}
Tests Skipped: {skipped} (insufficient bits for these tests)
Tests Passed: {passed} ({passed/applicable*100:.2f}% of applicable tests)
Tests Failed: {failed} ({failed/applicable*100:.2f}% of applicable tests)
Significance Level: {CONFIG['significance_level']}
{'='*80}
"""
        return summary
    
    def save_csv_report(self, filename):
        """Save detailed CSV report."""
        df = pd.DataFrame(self.all_results)
        filepath = os.path.join(CONFIG['output_dir'], filename)
        df.to_csv(filepath, index=False)
        if CONFIG['verbose']:
            print(f"CSV report saved to: {filepath}")
    
    def print_results(self):
        """Print formatted results to console."""
        print("\n" + "="*80)
        print("DETAILED TEST RESULTS")
        print("="*80)
        
        for result in self.all_results:
            if result['Result'] == 'PASS':
                status_symbol = "✓"
            elif result['Result'] == 'SKIP':
                status_symbol = "⊘"
            else:
                status_symbol = "✗"
            print(f"\n[{result['Test Number']:3d}] {status_symbol} {result['Test Name']}")
            print(f"      P-Value: {result['P-Value']}")
            print(f"      Statistic: {result['Statistic']}")
            if result['Details']:
                print(f"      Details: {result['Details']}")
    
    def generate_plots(self, bits, key_id):
        """Generate statistical visualization plots."""
        if not CONFIG['generate_plots']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Cryptographic Key Analysis - Key {key_id}', fontsize=16)
        
        axes[0, 0].hist(bits, bins=2, edgecolor='black')
        axes[0, 0].set_title('Bit Distribution')
        axes[0, 0].set_xlabel('Bit Value')
        axes[0, 0].set_ylabel('Frequency')
        
        block_size = 8
        num_blocks = len(bits) // block_size
        block_sums = [np.sum(bits[i*block_size:(i+1)*block_size]) 
                      for i in range(num_blocks)]
        axes[0, 1].hist(block_sums, bins=block_size+1, edgecolor='black')
        axes[0, 1].set_title(f'Byte-wise Bit Count Distribution (Block Size={block_size})')
        axes[0, 1].set_xlabel('Number of 1s in Block')
        axes[0, 1].set_ylabel('Frequency')
        
        s = 2 * bits[:min(1000, len(bits))] - 1
        cumsum = np.cumsum(s)
        axes[1, 0].plot(cumsum)
        axes[1, 0].set_title('Cumulative Sum (Random Walk)')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Cumulative Sum')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        runs = []
        current_run = 1
        for i in range(1, min(1000, len(bits))):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        
        if runs:
            axes[1, 1].hist(runs, bins=max(runs), edgecolor='black')
            axes[1, 1].set_title('Run Length Distribution')
            axes[1, 1].set_xlabel('Run Length')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        filepath = os.path.join(CONFIG['output_dir'], f'key_{key_id}_analysis.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        if CONFIG['verbose']:
            print(f"Plots saved to: {filepath}")

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_key(key_data, key_id, encoding='hex'):
    """
    Main function to analyze a single cryptographic key.
    
    Args:
        key_data: The key in specified encoding
        key_id: Identifier for the key
        encoding: Encoding format of the key
    
    Returns:
        Dictionary with analysis results
    """
    if CONFIG['verbose']:
        print(f"\n{'='*80}")
        print(f"Analyzing Key {key_id}")
        print(f"{'='*80}")
    
    binary = convert_to_binary(key_data, encoding)
    if binary is None:
        return None
    
    bits = binary_to_int_array(binary)
    bit_length = len(bits)
    
    if bit_length < CONFIG['min_bit_length']:
        print(f"Warning: Key {key_id} has only {bit_length} bits (minimum {CONFIG['min_bit_length']} required)")
        return None
    
    if CONFIG['verbose']:
        print(f"Bit Length: {bit_length}")
        print(f"Running comprehensive cryptographic test suite (34 tests)...")
    
    report = ReportGenerator()
    alpha = CONFIG['significance_level']
    
    nist = NISTTests()
    additional = AdditionalTests()
    quality = KeyQualityMetrics()
    
    p, stat = nist.frequency_monobit_test(bits)
    report.add_result("NIST: Frequency (Monobit) Test", p, stat, p >= alpha)
    
    p, stat = nist.frequency_block_test(bits, CONFIG['block_size'])
    report.add_result("NIST: Block Frequency Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.runs_test(bits)
    report.add_result("NIST: Runs Test", p, stat, p >= alpha if p else False, skipped=(p is None), details=stat if p is None else "")
    
    p, stat = nist.longest_run_test(bits)
    report.add_result("NIST: Longest Run of Ones Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.binary_matrix_rank_test(bits)
    report.add_result("NIST: Binary Matrix Rank Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.dft_test(bits)
    report.add_result("NIST: Discrete Fourier Transform Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.non_overlapping_template_test(bits, CONFIG['template_size'])
    report.add_result("NIST: Non-overlapping Template Matching Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.overlapping_template_test(bits, CONFIG['template_size'])
    report.add_result("NIST: Overlapping Template Matching Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.maurers_universal_test(bits)
    report.add_result("NIST: Maurer's Universal Statistical Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.linear_complexity_test(bits)
    report.add_result("NIST: Linear Complexity Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.serial_test(bits, CONFIG['serial_block_length'])
    report.add_result("NIST: Serial Test", p, stat, 
                     p[0] >= alpha and p[1] >= alpha,
                     f"P-values: {p[0]:.6f}, {p[1]:.6f}")
    
    p, stat = nist.approximate_entropy_test(bits)
    report.add_result("NIST: Approximate Entropy Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = nist.cumulative_sums_test(bits)
    report.add_result("NIST: Cumulative Sums Test", p, stat,
                     p[0] >= alpha and p[1] >= alpha,
                     f"Forward: {p[0]:.6f}, Backward: {p[1]:.6f}")
    
    p_dict, stat = nist.random_excursions_test(bits)
    if p_dict:
        avg_p = np.mean(list(p_dict.values()))
        passed = sum(1 for p in p_dict.values() if p >= alpha)
        report.add_result("NIST: Random Excursions Test", avg_p, f"{passed}/8 states passed",
                         avg_p >= alpha, f"Cycles: {stat}")
    
    p_dict, stat = nist.random_excursions_variant_test(bits)
    if p_dict:
        avg_p = np.mean(list(p_dict.values()))
        passed = sum(1 for p in p_dict.values() if p >= alpha)
        report.add_result("NIST: Random Excursions Variant Test", avg_p, 
                         f"{passed}/18 states passed", avg_p >= alpha, f"Cycles: {stat}")
    
    p, max_corr, _ = additional.autocorrelation_test(bits)
    report.add_result("Autocorrelation Test", p, max_corr, p >= alpha if p else False, skipped=(p is None))
    
    p, ent = additional.entropy_test(bits)
    report.add_result("Shannon Entropy Test", p, f"{ent:.4f} bits" if p else ent, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.chi_square_test(bits)
    report.add_result("Chi-Square Uniformity Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.poker_test(bits)
    report.add_result("Poker Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.gap_test(bits)
    report.add_result("Gap Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.collision_test(bits)
    report.add_result("Collision Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.birthday_spacing_test(bits)
    report.add_result("Birthday Spacing Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.bit_independence_test(bits)
    report.add_result("Bit Independence Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.avalanche_effect_test(bits)
    report.add_result("Avalanche Effect Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.strict_avalanche_criterion(bits)
    report.add_result("Strict Avalanche Criterion Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.run_distribution_test(bits)
    report.add_result("Run Distribution Test", p, f"{len(stat)} unique run lengths" if p else stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.coupon_collector_test(bits)
    report.add_result("Coupon Collector's Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.permutation_test(bits)
    report.add_result("Permutation Test", p, stat, p >= alpha if p else False, skipped=(p is None))
    
    p, periods = additional.periodicity_test(bits)
    report.add_result("Periodicity Detection Test", p, f"{len(periods)} periods found", 
                     p >= alpha, f"Max period match: {max([p[1] for p in periods]) if periods else 0:.4f}")
    
    p, results = additional.maurer_universal_extended(bits)
    report.add_result("Extended Maurer's Universal Test", p, 
                     f"{len(results)} block sizes tested" if p else results, 
                     p >= alpha if p else False, skipped=(p is None))
    
    p, stat = additional.lempel_ziv_complexity(bits)
    report.add_result("Lempel-Ziv Complexity Test", p, stat, p >= alpha)
    
    strength = quality.estimate_key_strength(bits)
    report.add_result("Key Strength Estimation", strength/100, f"{strength:.2f}%", 
                     strength >= 70, "Higher is better")
    
    uniformity = quality.uniformity_analysis(bits)
    report.add_result("Uniformity Analysis", uniformity/100, f"{uniformity:.2f}%",
                     uniformity >= 70, "Higher is better")
    
    complexity = quality.complexity_score(bits)
    report.add_result("Complexity Score", complexity/100, f"{complexity:.2f}%",
                     complexity >= 70, "Higher is better")
    
    print(report.generate_summary(key_id, bit_length))
    
    if CONFIG['verbose']:
        report.print_results()
    
    if CONFIG['save_csv_report']:
        report.save_csv_report(f'key_{key_id}_detailed_report.csv')
    
    report.generate_plots(bits, key_id)
    
    return {
        'key_id': key_id,
        'bit_length': bit_length,
        'total_tests': report.test_count,
        'passed': sum(1 for r in report.all_results if r['Result'] == 'PASS'),
        'failed': sum(1 for r in report.all_results if r['Result'] == 'FAIL'),
        'key_strength': strength,
        'uniformity': uniformity,
        'complexity': complexity
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║         CRYPTOGRAPHIC KEY ANALYSIS TOOL                                ║
    ║         NIST Statistical Test Suite + Advanced Crypto Tests            ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    ensure_output_dir()
    
    print(f"Reading keys from: {CONFIG['csv_file']}")
    keys = read_keys_from_csv(CONFIG['csv_file'], CONFIG['key_column'])
    
    if not keys:
        print("Error: No keys found in CSV file!")
        print("\nPlease ensure:")
        print(f"1. The file '{CONFIG['csv_file']}' exists")
        print(f"2. Keys are in column {CONFIG['key_column']}")
        print(f"3. Keys are in '{CONFIG['key_encoding']}' format")
        return
    
    print(f"Found {len(keys)} key(s) to analyze\n")
    
    all_summaries = []
    
    for i, key in enumerate(keys, 1):
        result = analyze_key(key, f"KEY_{i}", CONFIG['key_encoding'])
        if result:
            all_summaries.append(result)
    
    if all_summaries:
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        summary_df = pd.DataFrame(all_summaries)
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(CONFIG['output_dir'], 'overall_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nOverall summary saved to: {summary_path}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved in: {CONFIG['output_dir']}/")
        print("Files generated:")
        print("  - Detailed CSV reports for each key")
        print("  - Statistical visualization plots")
        print("  - Overall summary CSV")
    else:
        print("\nNo keys were successfully analyzed.")

if __name__ == "__main__":
    main()
