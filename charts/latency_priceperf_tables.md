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

## Throughput-match: v6e chips to match B200 (1 GPU) throughput

chips = ceil(GPU pairs/s / one-v6e pairs/s); v6e fleet $/hr = chips x $1.61. This shows the cost to match the GPU's aggregate throughput with TPU chips (note: adding chips scales throughput, not single-request latency).

| Batch | B200 (1 GPU) pairs/s | 1x v6e pairs/s | v6e chips to match | v6e fleet $/hr | B200 (1 GPU) $/hr | TPU fleet cheaper? |
|------:|-----------:|---------------:|-------------------:|---------------:|--------:|:--|
| 1 | 85.498 | 92.239 | 1 | $1.61 | $6.95 | yes |
| 2 | 116.606 | 100.221 | 2 | $3.22 | $6.95 | yes |
| 4 | 183.052 | 118.104 | 2 | $3.22 | $6.95 | yes |
| 8 | 282.684 | 155.212 | 2 | $3.22 | $6.95 | yes |
| 16 | 379.786 | 172.784 | 3 | $4.83 | $6.95 | yes |
| 32 | 438.642 | 182.667 | 3 | $4.83 | $6.95 | yes |
| 64 | 460.704 | 190.484 | 3 | $4.83 | $6.95 | yes |

## Throughput-match: v6e chips to match H200 (1 GPU) throughput

chips = ceil(GPU pairs/s / one-v6e pairs/s); v6e fleet $/hr = chips x $1.61. This shows the cost to match the GPU's aggregate throughput with TPU chips (note: adding chips scales throughput, not single-request latency).

| Batch | H200 (1 GPU) pairs/s | 1x v6e pairs/s | v6e chips to match | v6e fleet $/hr | H200 (1 GPU) $/hr | TPU fleet cheaper? |
|------:|-----------:|---------------:|-------------------:|---------------:|--------:|:--|
| 1 | 85.185 | 92.239 | 1 | $1.61 | $3.85 | yes |
| 2 | 98.105 | 100.221 | 1 | $1.61 | $3.85 | yes |
| 4 | 181.414 | 118.104 | 2 | $3.22 | $3.85 | yes |
| 8 | 262.667 | 155.212 | 2 | $3.22 | $3.85 | yes |
| 16 | 332.293 | 172.784 | 2 | $3.22 | $3.85 | yes |
| 32 | 362.625 | 182.667 | 2 | $3.22 | $3.85 | yes |
| 64 | 376.102 | 190.484 | 2 | $3.22 | $3.85 | yes |

## Throughput-match: v6e chips to match G4 RTX PRO 6000 (1 GPU) throughput

chips = ceil(GPU pairs/s / one-v6e pairs/s); v6e fleet $/hr = chips x $1.61. This shows the cost to match the GPU's aggregate throughput with TPU chips (note: adding chips scales throughput, not single-request latency).

| Batch | G4 RTX PRO 6000 (1 GPU) pairs/s | 1x v6e pairs/s | v6e chips to match | v6e fleet $/hr | G4 RTX PRO 6000 (1 GPU) $/hr | TPU fleet cheaper? |
|------:|-----------:|---------------:|-------------------:|---------------:|--------:|:--|
| 1 | 139.995 | 92.239 | 2 | $3.22 | $3.11 | no |
| 2 | 178.58 | 100.221 | 2 | $3.22 | $3.11 | no |
| 4 | 229.313 | 118.104 | 2 | $3.22 | $3.11 | no |
| 8 | 273.581 | 155.212 | 2 | $3.22 | $3.11 | no |
| 16 | 303.367 | 172.784 | 2 | $3.22 | $3.11 | no |
| 32 | 319.721 | 182.667 | 2 | $3.22 | $3.11 | no |
| 64 | 321.662 | 190.484 | 2 | $3.22 | $3.11 | no |

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

