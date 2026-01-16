"""
Experimente zum Vergleich der Algorithmen.

Vergleicht:
1. Boolean: Signatur-Methode (Algo 2.1) vs. Naive O(n³)
2. k-beschränkt: Schichten-Signaturen (Algo 4.1) vs. Strassen
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

from boolean_matrix_multiplier import BooleanMatrixMultiplier
from strassen_multiplier import StrassenMultiplier, KBoundedStrassenMultiplier
from k_bounded_multiplier import KBoundedMatrixMultiplier


def pure_python_naive_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Explizite O(n³) Matrixmultiplikation in reinem Python.
    Dies stellt einen fairen Vergleich zur Signatur-Methode dar.
    """
    n, k_dim = A.shape
    _, m = B.shape
    C = np.zeros((n, m), dtype=int)
    
    for i in range(n):
        for j in range(m):
            sum_val = 0
            for l in range(k_dim):
                # Standard Skalarprodukt-Berechnung
                sum_val += A[i, l] * B[l, j]
            C[i, j] = sum_val
    return C


def experiment_boolean_multiplication():
    """
    Experiment 1: Boolean Matrixmultiplikation.
    Vergleicht Algorithmus 2.1 (Signaturen O(n²)) mit Pure Python Naive O(n³).
    """
    print("="*80)
    print("EXPERIMENT 1: Boolean Matrixmultiplikation (FAIR COMPARISON)")
    print("="*80)
    
    # Reduzierte Größen für Pure Python Naive, da O(n³) in Python sehr langsam ist
    sizes = [8, 16, 32, 64, 128, 256] 
    
    times_signature = []
    times_naive = []
    
    multiplier_sig = BooleanMatrixMultiplier()
    
    for n in sizes:
        print(f"\nTeste Größe n={n}")
        np.random.seed(42)
        
        A = np.random.randint(0, 2, (n, n))
        B = np.random.randint(0, 2, (n, n))
        
        # Test 1: Signatur-Methode (Algo 2.1)
        start = time.perf_counter()
        C_sig = multiplier_sig.multiply_optimized(A, B)
        time_sig = time.perf_counter() - start
        times_signature.append(time_sig)
        print(f"  Signatur-Methode O(n²): {time_sig*1000:.3f} ms")
        
        # Test 2: Pure Python Naive O(n³)
        start = time.perf_counter()
        C_naive = pure_python_naive_multiply(A, B)
        # Für Boolean Vergleich: Werte > 0 zu 1 konvertieren
        C_naive = (C_naive > 0).astype(int)
        time_naive = time.perf_counter() - start
        times_naive.append(time_naive)
        print(f"  Pure Python Naive O(n³): {time_naive*1000:.3f} ms")
            
        # Verifikation
        assert np.array_equal(C_sig, C_naive), f"Fehler bei n={n}: Signatur != Naive"
    
    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, [t*1000 for t in times_signature], 'o-', 
             label='Signatur-Methode O(n²)', linewidth=2, markersize=8)
    plt.plot(sizes, [t*1000 for t in times_naive], 's-', 
             label='Pure Python Naive O(n³)', linewidth=2, markersize=8)
    
    plt.xlabel('Matrixgröße n', fontsize=12)
    plt.ylabel('Laufzeit (ms)', fontsize=12)
    plt.title('Fairer Vergleich: O(n²) Signaturen vs. O(n³) Naive (Rein Python)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    
    output_path = Path(__file__).parent / 'experiment_boolean_fair.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"\n✓ Plot gespeichert: {output_path}")
    plt.close()
    
    return sizes, times_signature, times_naive


def experiment_k_bounded_multiplication():
    """
    Experiment 2: k-beschränkte Matrixmultiplikation.
    
    Vergleicht Algorithmus 4.1 (Schichten-Signaturen O(k·n²)) mit Strassen O(n^2.807).
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: k-beschränkte Matrixmultiplikation")
    print("="*80)
    
    k_values = [2, 3, 5]
    sizes = [8, 16, 32, 64, 128]
    
    results = {}
    
    multiplier_sig = KBoundedMatrixMultiplier()
    multiplier_strassen = KBoundedStrassenMultiplier(threshold=32)
    
    for k in k_values:
        print(f"\n--- k = {k} ---")
        times_sig = []
        times_strassen = []
        
        for n in sizes:  # Diese Schleife MUSS eingerückt sein!
            print(f"  Teste Größe n={n}")
            np.random.seed(42)
            
            # Generiere zufällige k-beschränkte Matrizen
            A = np.random.randint(0, k + 1, (n, n))
            B = np.random.randint(0, k + 1, (n, n))
            
            # Test 1: Schichten-Signaturen (Algo 4.1)
            start = time.perf_counter()
            C_sig = multiplier_sig.multiply_k_bounded(A, B, k)
            time_sig = time.perf_counter() - start
            times_sig.append(time_sig)
            print(f"    Schichten-Signaturen: {time_sig*1000:.3f} ms")
            
            # Test 2: Strassen
            start = time.perf_counter()
            C_strassen = multiplier_strassen.multiply(A, B)
            time_strassen = time.perf_counter() - start
            times_strassen.append(time_strassen)
            print(f"    Strassen:             {time_strassen*1000:.3f} ms")
            
            # Verifikation
            expected = np.matmul(A, B)
            assert np.array_equal(C_sig, expected), f"Fehler bei k={k}, n={n}: Signatur != NumPy"
            assert np.array_equal(C_strassen, expected), f"Fehler bei k={k}, n={n}: Strassen != NumPy"
        
        results[k] = (times_sig, times_strassen)
    
    # Plot erstellen
    plt.figure(figsize=(12, 6))
    
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, k in enumerate(k_values):
        times_sig, times_strassen = results[k]
        
        plt.plot(sizes, [t*1000 for t in times_sig], 
                marker=markers[idx], linestyle='-', color=colors[idx],
                label=f'Schichten-Signaturen (k={k})', linewidth=2, markersize=8)
        plt.plot(sizes, [t*1000 for t in times_strassen], 
                marker=markers[idx], linestyle='--', color=colors[idx],
                label=f'Strassen (k={k})', linewidth=2, alpha=0.7, markersize=8)
    
    plt.xlabel('Matrixgröße n', fontsize=12)
    plt.ylabel('Laufzeit (ms)', fontsize=12)
    plt.title('k-beschränkte Matrixmultiplikation: Schichten-Signaturen vs. Strassen', fontsize=14)
    plt.legend(fontsize=9, ncol=2, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log', base=2)
    
    # Speichere als SVG
    output_path = Path(__file__).parent / 'experiment_k_bounded.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    print(f"\n✓ Plot gespeichert: {output_path}")
    plt.close()
    
    return results, sizes


def print_summary(exp1_data, exp2_data):
    """
    Druckt eine Zusammenfassung der Experimente und berechnet Speedups.
    Diese Version ist sicher gegen Index-Fehler (ValueError).
    """
    # Entpacken der Daten aus Experiment 1 (Boolean)
    sizes_bool, times_sig, times_naive = exp1_data
    
    # Entpacken der Daten aus Experiment 2 (k-beschränkt)
    # HINWEIS: Falls experiment_k_bounded_multiplication nur 'results' liefert,
    # stelle sicher, dass es (results, sizes) zurückgibt.
    results_k, sizes_k = exp2_data
    
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG")
    print("="*80)
    
    print("\n1. Boolean Matrixmultiplikation:")
    # Wir prüfen dynamisch auf n=256 und n=512
    for target_n in [256, 512]:
        if target_n in sizes_bool:
            idx = sizes_bool.index(target_n)
            # Sicherstellen, dass für diesen Index auch ein Naive-Wert existiert
            if idx < len(times_naive) and times_naive[idx] is not None:
                speedup = times_naive[idx] / times_sig[idx]
                print(f"   Größe n={target_n}:")
                print(f"     Signatur-Methode O(n²): {times_sig[idx]*1000:.2f} ms")
                print(f"     Naive O(n³) (Python):   {times_naive[idx]*1000:.2f} ms")
                print(f"     Speedup:                {speedup:.2f}x")
        else:
            print(f"   Größe n={target_n}: (Nicht in den Experimenten enthalten)")
    
    print("\n2. k-beschränkte Matrixmultiplikation:")
    # Nutzt die letzte (größte) gemessene Größe aus dem Experiment
    last_n = sizes_k[-1] if sizes_k else "unbekannt"
    print(f"   Vergleich bei maximaler Größe n={last_n}:")
    
    for k, (t_sig_list, t_strassen_list) in results_k.items():
        if t_sig_list and t_strassen_list:
            t_sig = t_sig_list[-1]
            t_strassen = t_strassen_list[-1]
            ratio = t_sig / t_strassen
            print(f"   k={k}:")
            print(f"     Schichten-Signaturen O(k·n²): {t_sig*1000:.2f} ms")
            print(f"     Strassen O(n^2.807):          {t_strassen*1000:.2f} ms")
            print(f"     Verhältnis (Sig/Strassen):    {ratio:.2f}x")
    
    print("\n" + "="*80)
    print("ERKENNTNISSE")
    print("="*80)
    print("\n1. Boolean Matrixmultiplikation:")
    print("   - Die Signatur-Methode zeigt einen massiven Vorteil durch O(n²).")
    print("   - Da beide in reinem Python laufen, ist der Speedup-Faktor rein algorithmisch.")
    
    print("\n2. k-beschränkte Matrixmultiplikation:")
    print("   - Schichten-Signaturen skalieren linear mit k.")
    print("   - Strassen bleibt bei kleinen k oft überlegen, da die Bit-Operationen in")
    print("     Python bei sehr großen Signaturen (großes n) Overhead erzeugen.")


if __name__ == "__main__":
    print("Starte faire Experimente (Rein Python Vergleich)...")
    
    # Experiment 1
    exp1_data = experiment_boolean_multiplication()
    
    # Experiment 2 (nimmt nun die Rückgabe der Größen auf)
    exp2_data = experiment_k_bounded_multiplication()
    
    # Zusammenfassung aufrufen
    print_summary(exp1_data, exp2_data)
    
    print("\n✓ Alle Experimente abgeschlossen!")
    print("\nGenerierte Plots:")
    print("  - experiments/experiment_boolean.svg")
    print("  - experiments/experiment_k_bounded.svg")