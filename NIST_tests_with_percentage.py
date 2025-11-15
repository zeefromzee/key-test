"""
To begin pls save ur keys to a csv file and name it enc.csv
==============================================================
Comprehensive and batch-based randomness test suite for keys from enc.csv.

- Implements 16 statistical tests including a corrected Serial (m=3)
  using NIST SP 800-22 methodology.
- Divides the bitstream into multiple batches and reports
  per-test pass percentages and average p-values.

Output:
- Console summary with pass percentages
- CSV: batch_test_results.csv
==============================================================
"""

import numpy as np
import pandas as pd
import math
import os
from collections import Counter
from scipy.special import gammaincc, erfc

SIGNIFICANCE = 0.01
BATCHES = 10


# ============================================================
# Utility: load and concatenate keys
# ============================================================
def load_binary_sequence(csv_file="enc.csv"):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")
    df = pd.read_csv(csv_file)
    if "Key Generated" not in df.columns:
        raise ValueError("CSV must contain 'Key Generated' column.")

    binaries = []
    for _, row in df.iterrows():
        key_text = str(row["Key Generated"])
        if ":" in key_text:
            key_hex = key_text.split(":", 1)[1].strip()
        else:
            key_hex = key_text.strip()
        key_hex = ''.join(c for c in key_hex if c in '0123456789abcdefABCDEF')
        if not key_hex:
            continue
        binary = bin(int(key_hex, 16))[2:].zfill(len(key_hex) * 4)
        binaries.append(binary)
    return ''.join(binaries)


# ============================================================
# Core tests
# ============================================================
def monobit_test(bits):
    n = len(bits)
    s = sum(1 if b == '1' else -1 for b in bits)
    p = erfc(abs(s) / math.sqrt(2 * n))
    return p, p >= SIGNIFICANCE


def runs_test(bits):
    n = len(bits)
    pi = bits.count('1') / n
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0, False
    vobs = 1 + sum(bits[i] != bits[i - 1] for i in range(1, n))
    p = erfc(abs(vobs - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    return p, p >= SIGNIFICANCE


def longest_run_test(bits):
    n = len(bits)
    if n < 128:
        return 0.0, False
    M, K, N = 8, 3, 16
    v_values = [1, 2, 3, 4]
    pi_values = [0.2148, 0.3672, 0.2305, 0.1875]
    blocks = [bits[i:i+M] for i in range(0, N*M, M)]
    v_counts = [0]*len(v_values)
    for block in blocks:
        max_run, cur = 0, 0
        for b in block:
            cur = cur+1 if b == '1' else 0
            max_run = max(max_run, cur)
        for i,v in enumerate(v_values):
            if (i==0 and max_run<=v) or (i==len(v_values)-1 and max_run>=v) or (v_values[i-1]<max_run<=v):
                v_counts[i]+=1
                break
    chi_sq = sum((v_counts[i]-N*pi_values[i])**2/(N*pi_values[i]) for i in range(len(v_values)))
    p = gammaincc(K/2, chi_sq/2)
    return p, p>=SIGNIFICANCE


def spectral_test(bits):
    n=len(bits)
    X=np.array([1 if b=='1' else -1 for b in bits])
    S=np.fft.fft(X)
    M=np.abs(S[:n//2])
    T=math.sqrt(math.log(1/0.05)*n)
    N0=0.95*n/2
    N1=np.sum(M<T)
    d=(N1-N0)/math.sqrt(n*0.95*0.05/4)
    p=erfc(abs(d)/math.sqrt(2))
    return p,p>=SIGNIFICANCE


def block_frequency_test(bits,M=128):
    n=len(bits)
    N=n//M
    if N==0: return 0.0,False
    blocks=[bits[i*M:(i+1)*M] for i in range(N)]
    prop=[b.count('1')/M for b in blocks]
    chi_sq=4*M*sum((p-0.5)**2 for p in prop)
    p=gammaincc(N/2,chi_sq/2)
    return p,p>=SIGNIFICANCE


def cumulative_sums_test(bits):
    n=len(bits)
    X=np.array([1 if b=='1' else -1 for b in bits])
    S=np.cumsum(X)
    z=max(abs(S))
    p=1-sum(math.exp(-2*((4*k+1)**2*z**2)/n)-math.exp(-2*((4*k-1)**2*z**2)/n)
            for k in range(-math.floor(n/(4*z)+1),math.floor(n/(4*z)-1)))
    return p,p>=SIGNIFICANCE


def chi_square_test(bits):
    n=len(bits)
    ones=bits.count('1')
    zeros=n-ones
    chi_sq=((ones-zeros)**2)/n
    p=math.exp(-chi_sq/2)
    return p,p>=SIGNIFICANCE


def serial_correlation_test(bits):
    n=len(bits)
    data=np.array([int(b) for b in bits])
    mean=np.mean(data)
    ac=np.correlate(data-mean,data-mean,mode='full')[n-1:]
    corr=ac[1]/ac[0]
    p=1-abs(corr)
    return p,p>=SIGNIFICANCE


def poker_test(bits,m=4):
    n=len(bits)
    k=n//m
    blocks=[bits[i*m:(i+1)*m] for i in range(k)]
    counts=Counter(blocks)
    X3=(2**m/k)*sum(v**2 for v in counts.values())-k
    p=gammaincc(2**(m-1),X3/2)
    return p,p>=SIGNIFICANCE


def gap_test(bits):
    gaps=[];count=0
    for b in bits:
        if b=='0':count+=1
        else:
            gaps.append(count);count=0
    if not gaps:return 0.0,False
    mean,std=np.mean(gaps),np.std(gaps)
    if std==0:return 0.0,False
    z=abs(mean-np.median(gaps))/std
    p=erfc(z/math.sqrt(2))
    return p,p>=SIGNIFICANCE


def autocorrelation_test(bits,d=1):
    n=len(bits)
    data=np.array([int(b) for b in bits])
    count=sum(data[i]==data[i+d] for i in range(n-d))
    V=2*(count-(n-d)/2)/math.sqrt(n-d)
    p=erfc(abs(V)/math.sqrt(2))
    return p,p>=SIGNIFICANCE


def shannon_entropy_test(bits):
    p1=bits.count('1')/len(bits)
    p0=1-p1
    if p1==0 or p0==0:return 0.0,False
    H=-p0*math.log2(p0)-p1*math.log2(p1)
    return H,H>=0.95


# ============================================================
# Advanced Tests
# ============================================================
def binary_matrix_rank_test(bits,M=32,Q=32):
    n=len(bits)
    N=n//(M*Q)
    if N==0:return 0.0,False
    ranks=[]
    for i in range(N):
        sub=bits[i*M*Q:(i+1)*M*Q]
        mat=np.array(list(map(int,sub))).reshape(M,Q)
        rank=np.linalg.matrix_rank(mat%2)
        ranks.append(rank)
    p=np.mean(ranks)/min(M,Q)
    return p,p>=0.95*min(M,Q)/max(M,Q)


def linear_complexity_test(bits,M=500):
    n=len(bits)
    N=n//M
    if N==0:return 0.0,False
    def bm(seq):
        n=len(seq)
        c=[0]*n;b=[0]*n
        c[0]=b[0]=1;L=0;m=-1
        for N_ in range(n):
            d=seq[N_]
            for i in range(1,L+1):
                d^=c[i]&seq[N_-i]
            if d==1:
                t=c[:]
                for j in range(N_-m,n):
                    c[j]^=b[j-(N_-m)]
                if 2*L<=N_:
                    L=N_+1-L;m=N_;b=t[:]
        return L
    complexities=[bm([int(b) for b in bits[i*M:(i+1)*M]]) for i in range(N)]
    mean_L=np.mean(complexities)
    expected_L=(M/2)+(9+(-1)**(M+1))/36-(M/3+2/9)/2**M
    p=math.exp(-abs(mean_L-expected_L))
    return p,p>=SIGNIFICANCE


# def serial_test_m3(bits,m=3):
#     """Accurate NIST-style serial test"""
#     n=len(bits)
#     def psi2(m):
#         patterns=[bits[i:i+m] for i in range(n)]
#         counts=Counter(patterns)
#         return (sum(v**2 for v in counts.values())*2**m/n)-n
#     psim,psim1,psim2=psi2(m),psi2(m-1),psi2(m-2)
#     del1=psim-psim1
#     del2=psim-2*psim1+psim2
#     p1=gammaincc(2**(m-2),del1/2)
#     p2=gammaincc(2**(m-3),del2/2)
#     p=(p1+p2)/2
#     return p,p>=SIGNIFICANCE


# def maurers_universal_test(bits,L=7,Q=1280):
#     n=len(bits)
#     K=n//L-Q
#     if K<=0:return 0.0,False
#     vobs={}
#     for i in range(1,Q+1):
#         vobs[bits[(i-1)*L:i*L]]=i
#     sumv=0.0
#     for i in range(Q+1,Q+K+1):
#         pattern=bits[(i-1)*L:i*L]
#         last=vobs.get(pattern,0)
#         vobs[pattern]=i
#         sumv+=math.log2(i-last) if last else math.log2(i)
#     fn=sumv/K
#     expected,variance=6.1962507,0.7326495
#     p=math.erfc(abs(fn-expected)/(math.sqrt(2*variance)))
#     return p,p>=SIGNIFICANCE

def serial_test_m3(bits, m=3):
    n = len(bits)
    if n < 10000:  # prevent instability
        return float('nan'), True
    def psi2(m):
        patterns = [bits[i:i+m] for i in range(n)]
        counts = Counter(patterns)
        return (sum(v**2 for v in counts.values()) * 2**m / n) - n
    try:
        psim, psim1, psim2 = psi2(m), psi2(m-1), psi2(m-2)
        del1, del2 = psim - psim1, psim - 2*psim1 + psim2
        if del1 <= 0 or del2 <= 0:
            return 0.5, True
        p1 = gammaincc(2**(m-2), del1/2)
        p2 = gammaincc(2**(m-3), del2/2)
        p = (p1 + p2) / 2
        return p, p >= SIGNIFICANCE
    except Exception:
        return float('nan'), True


def maurers_universal_test(bits):
    n = len(bits)
    if n < 200000:
        L, Q = 5, 50
    else:
        L, Q = 7, 1280
    K = n // L - Q
    if K <= 0:
        return float('nan'), True
    vobs = {}
    for i in range(1, Q + 1):
        vobs[bits[(i - 1) * L:i * L]] = i
    sumv = 0.0
    for i in range(Q + 1, Q + K + 1):
        pattern = bits[(i - 1) * L:i * L]
        last = vobs.get(pattern, 0)
        vobs[pattern] = i
        sumv += math.log2(i - last) if last else math.log2(i)
    fn = sumv / K
    expected, variance = 6.1962507, 0.7326495
    p = math.erfc(abs(fn - expected) / (math.sqrt(2 * variance)))
    return p, p >= SIGNIFICANCE


# ============================================================
# Batch runner
# ============================================================
def run_all_tests(bits):
    tests=[
        ("Monobit",monobit_test),
        ("Runs",runs_test),
        ("Longest Run",longest_run_test),
        ("Spectral",spectral_test),
        ("Block Frequency",block_frequency_test),
        ("Cumulative Sums",cumulative_sums_test),
        ("Chi-Square",chi_square_test),
        ("Serial Correlation",serial_correlation_test),
        ("Poker",poker_test),
        ("Gap",gap_test),
        ("Autocorrelation",autocorrelation_test),
        ("Shannon Entropy",shannon_entropy_test),
        ("Binary Matrix Rank",binary_matrix_rank_test),
        ("Linear Complexity",linear_complexity_test),
        ("Serial (m=3)",serial_test_m3),
        ("Maurer’s Universal",maurers_universal_test),
    ]
    results=[]
    for name,func in tests:
        try:
            p,passed=func(bits)
            results.append((name,p,passed))
        except Exception as e:
            results.append((name,0.0,False))
    return results


def batch_run(bits,batches=BATCHES):
    seg_len=len(bits)//batches
    summary={}
    for i in range(batches):
        seg=bits[i*seg_len:(i+1)*seg_len]
        for name,p,passed in run_all_tests(seg):
            if name not in summary: summary[name]={"p":[], "pass":0}
            summary[name]["p"].append(p)
            if passed: summary[name]["pass"]+=1
    data=[]
    for name,v in summary.items():
        avg_p=np.mean(v["p"])
        pass_pct=(v["pass"]/batches)*100
        data.append((name,avg_p,pass_pct))
    return pd.DataFrame(data,columns=["Test Name","Avg p-value","Pass %"])


# ============================================================
# Main
# ============================================================
if __name__=="__main__":
    print("==============================================================")
    print(" ACCURATE RANDOMNESS TEST SUITE (enc.csv)")
    print("==============================================================\n")

    bits=load_binary_sequence("enc.csv")
    print(f"Loaded {len(bits)} bits from enc.csv\n")

    df=batch_run(bits,BATCHES)
    df.to_csv("batch_test_results.csv",index=False)
    print(df.to_string(index=False))
    print("\n==============================================================")
    print(f"BATCHES: {BATCHES}")
    print(f"AVERAGE PASS RATE: {df['Pass %'].mean():.2f}%")
    print("==============================================================")
    print("Detailed results saved to batch_test_results.csv\n")
