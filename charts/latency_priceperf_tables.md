# Latency-first price/performance across accelerators

Real-time serving; prefix caching OFF; synthetic seq-len 512; fresh content per request (no KV/prefix reuse). Prices per chip-hour: TPU v6e $1.61, B200 $6.95, H200 $3.85, G4 $3.11.

## Request latency p50/p99 (ms) by batch size

| Batch | TPU v6e (torchax) p50 | B200 (1 GPU) p50 | H200 (1 GPU) p50 | G4 RTX PRO 6000 (1 GPU) p50 |
|------:|---:|---:|---:|---:|
| 1 | 10.793 | 11.451 | 11.533 | 7.104 |
| 2 | 20.423 | 19.653 | 20.153 | 9.221 |
| 4 | 31.673 | 23.388 | 23.098 | 18.122 |
| 8 | 50.295 | 28.72 | 30.696 | 29.281 |
| 16 | 92.316 | 42.255 | 47.73 | 52.677 |
| 32 | 173.995 | 72.12 | 87.551 | 100.025 |
| 64 | 335.344 | 138.802 | 169.198 | 198.313 |

## Throughput (pairs/s) by batch size

| Batch | TPU v6e (torchax) pairs/s | B200 (1 GPU) pairs/s | H200 (1 GPU) pairs/s | G4 RTX PRO 6000 (1 GPU) pairs/s |
|------:|---:|---:|---:|---:|
| 1 | 92.239 | 85.498 | 85.185 | 139.995 |
| 2 | 100.221 | 116.606 | 98.105 | 178.58 |
| 4 | 118.104 | 183.052 | 181.414 | 229.313 |
| 8 | 155.212 | 282.684 | 262.667 | 273.581 |
| 16 | 172.784 | 379.786 | 332.293 | 303.367 |
| 32 | 182.667 | 438.642 | 362.625 | 319.721 |
| 64 | 190.484 | 460.704 | 376.102 | 321.662 |

## Both price/perf views at batch size 32 (customer operating point)

| Device | p50 latency (ms) | p99 latency (ms) | pairs/s | $/1M pairs (throughput) | $/1000 req (latency) |
|---|---:|---:|---:|---:|---:|
| TPU v6e (torchax) | 173.995 | 187.747 | 182.667 | $2.4483 | $0.07781 |
| B200 (1 GPU) | 72.12 | 78.676 | 438.642 | $4.4012 | $0.13923 |
| H200 (1 GPU) | 87.551 | 98.34 | 362.625 | $2.9492 | $0.09363 |
| G4 RTX PRO 6000 (1 GPU) | 100.025 | 101.605 | 319.721 | $2.702 | $0.08641 |

## Latency-match: TPU v6e meeting B200 (1 GPU) request latency

For each GPU batch size we take its p50 request latency, then find the fastest v6e config (largest batch) whose p50 is still <= that GPU latency, i.e. the v6e can serve within the same latency budget. We then compare cost per 1000 requests at that budget.

| GPU batch | B200 (1 GPU) p50 (ms) | B200 (1 GPU) $/1k req | v6e config that meets it | v6e p50 (ms) | v6e $/1k req | Cheaper at equal latency |
|------:|---:|---:|:--|---:|---:|:--|
| 1 | 11.451 | $0.02211 | v6e bs1 | 10.793 | $0.00483 | TPU |
| 2 | 19.653 | $0.03794 | v6e bs1 | 10.793 | $0.00483 | TPU |
| 4 | 23.388 | $0.04515 | v6e bs2 | 20.423 | $0.00913 | TPU |
| 8 | 28.72 | $0.05545 | v6e bs2 | 20.423 | $0.00913 | TPU |
| 16 | 42.255 | $0.08158 | v6e bs4 | 31.673 | $0.01416 | TPU |
| 32 | 72.12 | $0.13923 | v6e bs8 | 50.295 | $0.02249 | TPU |
| 64 | 138.802 | $0.26796 | v6e bs16 | 92.316 | $0.04129 | TPU |

## Latency-match: TPU v6e meeting H200 (1 GPU) request latency

For each GPU batch size we take its p50 request latency, then find the fastest v6e config (largest batch) whose p50 is still <= that GPU latency, i.e. the v6e can serve within the same latency budget. We then compare cost per 1000 requests at that budget.

| GPU batch | H200 (1 GPU) p50 (ms) | H200 (1 GPU) $/1k req | v6e config that meets it | v6e p50 (ms) | v6e $/1k req | Cheaper at equal latency |
|------:|---:|---:|:--|---:|---:|:--|
| 1 | 11.533 | $0.01233 | v6e bs1 | 10.793 | $0.00483 | TPU |
| 2 | 20.153 | $0.02155 | v6e bs1 | 10.793 | $0.00483 | TPU |
| 4 | 23.098 | $0.0247 | v6e bs2 | 20.423 | $0.00913 | TPU |
| 8 | 30.696 | $0.03283 | v6e bs2 | 20.423 | $0.00913 | TPU |
| 16 | 47.73 | $0.05104 | v6e bs4 | 31.673 | $0.01416 | TPU |
| 32 | 87.551 | $0.09363 | v6e bs8 | 50.295 | $0.02249 | TPU |
| 64 | 169.198 | $0.18095 | v6e bs16 | 92.316 | $0.04129 | TPU |

## Latency-match: TPU v6e meeting G4 RTX PRO 6000 (1 GPU) request latency

For each GPU batch size we take its p50 request latency, then find the fastest v6e config (largest batch) whose p50 is still <= that GPU latency, i.e. the v6e can serve within the same latency budget. We then compare cost per 1000 requests at that budget.

| GPU batch | G4 RTX PRO 6000 (1 GPU) p50 (ms) | G4 RTX PRO 6000 (1 GPU) $/1k req | v6e config that meets it | v6e p50 (ms) | v6e $/1k req | Cheaper at equal latency |
|------:|---:|---:|:--|---:|---:|:--|
| 1 | 7.104 | $0.00614 | none (even v6e bs1 10.793 ms is slower) | - | - | G4 RTX PRO 6000 (1 GPU) (TPU cannot match this latency) |
| 2 | 9.221 | $0.00797 | none (even v6e bs1 10.793 ms is slower) | - | - | G4 RTX PRO 6000 (1 GPU) (TPU cannot match this latency) |
| 4 | 18.122 | $0.01566 | v6e bs1 | 10.793 | $0.00483 | TPU |
| 8 | 29.281 | $0.0253 | v6e bs2 | 20.423 | $0.00913 | TPU |
| 16 | 52.677 | $0.04551 | v6e bs8 | 50.295 | $0.02249 | TPU |
| 32 | 100.025 | $0.08641 | v6e bs16 | 92.316 | $0.04129 | TPU |
| 64 | 198.313 | $0.17132 | v6e bs32 | 173.995 | $0.07781 | TPU |

