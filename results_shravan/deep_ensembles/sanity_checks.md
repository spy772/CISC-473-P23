# Sanity checks

- PASS Ensemble M=5 vs baseline: NLL 0.2232 vs 0.3026, ECE 0.0164 vs 0.0312
- PASS Ensemble accuracy 0.9273 vs baseline 0.9028
- **FLAG** Member accuracy spread: 0.8979-0.9077 (range 0.0097, limit 0.005)
- PASS baseline (seed 0, Edward ckpt): fixed split coverage 0.8959 (0.88-0.93)
- PASS member seed 1: fixed split coverage 0.8965 (0.88-0.93)
- PASS member seed 2: fixed split coverage 0.8988 (0.88-0.93)
- PASS member seed 3: fixed split coverage 0.8978 (0.88-0.93)
- PASS member seed 4: fixed split coverage 0.8973 (0.88-0.93)
- PASS ensemble M=2: fixed split coverage 0.8935 (0.88-0.93)
- PASS ensemble M=3: fixed split coverage 0.8945 (0.88-0.93)
- PASS ensemble M=4: fixed split coverage 0.8922 (0.88-0.93)
- PASS ensemble M=5: fixed split coverage 0.8928 (0.88-0.93)
