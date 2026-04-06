# Actual vs Estimate - qwen3:1.7b (Vulkan, Intel Arc)

Date: 2026-04-06
Model: qwen3:1.7b (num_heads=16, num_kv_heads=4, head_dim=128, sliding_window=256)
Flash Attention: ON
Each input length tested twice, results averaged. Run 1 of input_length=18 excluded (cache effect).

## Summary

| Input Length | Actual Batch | Prefill GPU (ms) | Est Prefill (ms) | Ratio | Decode GPU/tok (ms) | Est Decode (ms) | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 18 | 410.9 | 225.5 | **0.55x** | 81.3 | 71.1 | **0.87x** |
| 150 | 153 | 1638.3 | 701.9 | **0.43x** | 117.2 | 71.1 | **0.61x** |
| 300 | 325 | 4080.2 | 1175.1 | **0.29x** | 131.9 | 77.8 | **0.59x** |

## API Timing

| Input Length | prompt_eval_count | Run 1 prefill (ms) | Run 2 prefill (ms) | eval_count | Run 1 decode (ms) | Run 2 decode (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 18 | 21 | 414.6 | 450.2 | 20 | 1690.4 | 1755.3 |
| 150 | 156 | 1,640.5 | 1,667.0 | 5 | 512.3 | 502.7 |
| 300 | 328 | 4,210.6 | 4,044.7 | 5 | 610.1 | 505.6 |

### input_length=18: Prefill (batch=18)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| MUL_MAT q4_K | 254,451 | 123,345 | 0.48x | 61.9% | 54.7% |
| MUL_MAT q6_K | 81,238 | 36,171 | 0.45x | 19.8% | 16.0% |
| MUL_MAT f16 | 36,719 | 26,637 | 0.73x | 8.9% | 11.8% |
| FLASH_ATTN_EXT | 26,932 | 21,538 | 0.80x | 6.6% | 9.6% |
| RMS_NORM_MUL_ROPE | 4,619 | 5,238 | 1.13x | 1.1% | 2.3% |
| GLU | 2,482 | 501 | 0.20x | 0.6% | 0.2% |
| RMS_NORM_MUL | 2,456 | 5,371 | 2.19x | 0.6% | 2.4% |
| ADD | 1,044 | 2,355 | 2.26x | 0.3% | 1.0% |
| SET_ROWS | 523 | 1,228 | 2.35x | 0.1% | 0.5% |
| MUL_MAT_ADD q6_K | 403 | 311 | 0.77x | 0.1% | 0.1% |
| GET_ROWS | 14 | 0 | 0.00x | 0.0% | - |
| CPY | 8 | 0 | 0.0x | 0.0% | - |
| **Total** | **410,888** | **225,503** | **0.55x** | | |

### input_length=18: Decode (per token)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| MUL_MAT q4_K | 32,615 | 27,761 | 0.85x | 40.1% | 39.0% |
| FLASH_ATTN_EXT | 13,148 | 7,396 | 0.56x | 16.2% | 10.4% |
| MUL_MAT q6_K | 12,520 | 8,761 | 0.70x | 15.4% | 12.3% |
| MUL_MAT_ADD q4_K | 11,359 | 8,726 | 0.77x | 14.0% | 12.3% |
| MUL_MAT_ADD q6_K | 5,502 | 4,351 | 0.79x | 6.8% | 6.1% |
| MUL_MAT f16 | 3,791 | 4,177 | 1.10x | 4.7% | 5.9% |
| RMS_NORM_MUL | 1,080 | 3,205 | 2.97x | 1.3% | 4.5% |
| RMS_NORM_MUL_ROPE | 738 | 2,670 | 3.62x | 0.9% | 3.8% |
| SET_ROWS | 359 | 1,228 | 3.42x | 0.4% | 1.7% |
| GLU | 179 | 0 | 0.00x | 0.2% | - |
| GET_ROWS | 17 | 0 | 0.00x | 0.0% | - |
| ADD | 15 | 33 | 2.28x | 0.0% | 0.0% |
| CPY | 6 | 0 | 0.0x | 0.0% | - |
| **Total** | **81,328** | **71,146** | **0.87x** | | |

### input_length=150: Prefill (batch=153)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| MUL_MAT q4_K | 738,120 | 399,933 | 0.54x | 45.1% | 57.0% |
| FLASH_ATTN_EXT | 478,352 | 47,525 | 0.10x | 29.2% | 6.8% |
| MUL_MAT q6_K | 244,508 | 116,773 | 0.48x | 14.9% | 16.6% |
| MUL_MAT f16 | 91,917 | 66,148 | 0.72x | 5.6% | 9.4% |
| RMS_NORM_MUL_ROPE | 42,280 | 26,282 | 0.62x | 2.6% | 3.7% |
| ADD | 14,557 | 6,585 | 0.45x | 0.9% | 0.9% |
| RMS_NORM_MUL | 14,507 | 30,092 | 2.07x | 0.9% | 4.3% |
| GLU | 11,396 | 4,247 | 0.37x | 0.7% | 0.6% |
| SET_ROWS | 2,129 | 1,228 | 0.58x | 0.1% | 0.2% |
| MUL_MAT_ADD q6_K | 463 | 311 | 0.67x | 0.0% | 0.0% |
| GET_ROWS | 15 | 0 | 0.00x | 0.0% | - |
| CPY | 14 | 0 | 0.00x | 0.0% | - |
| **Total** | **1,638,260** | **701,936** | **0.43x** | | |

### input_length=150: Decode (per token)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| MUL_MAT q4_K | 44,878 | 27,761 | 0.62x | 38.3% | 39.0% |
| FLASH_ATTN_EXT | 26,357 | 7,396 | 0.28x | 22.5% | 10.4% |
| MUL_MAT q6_K | 17,597 | 8,761 | 0.50x | 15.0% | 12.3% |
| MUL_MAT_ADD q4_K | 12,207 | 8,726 | 0.71x | 10.4% | 12.3% |
| MUL_MAT_ADD q6_K | 8,226 | 4,351 | 0.53x | 7.0% | 6.1% |
| MUL_MAT f16 | 4,704 | 4,177 | 0.89x | 4.0% | 5.9% |
| RMS_NORM_MUL_ROPE | 1,283 | 2,670 | 2.08x | 1.1% | 3.8% |
| RMS_NORM_MUL | 1,281 | 3,205 | 2.50x | 1.1% | 4.5% |
| SET_ROWS | 402 | 1,228 | 3.05x | 0.3% | 1.7% |
| GLU | 226 | 0 | 0.00x | 0.2% | - |
| GET_ROWS | 15 | 0 | 0.00x | 0.0% | - |
| ADD | 11 | 33 | 2.92x | 0.0% | 0.0% |
| CPY | 5 | 0 | 0.0x | 0.0% | - |
| **Total** | **117,193** | **71,146** | **0.61x** | | |

### input_length=300: Prefill (batch=325)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| FLASH_ATTN_EXT | 1,633,969 | 137,877 | 0.08x | 40.0% | 11.7% |
| MUL_MAT q4_K | 1,613,496 | 629,940 | 0.39x | 39.5% | 53.6% |
| MUL_MAT q6_K | 441,170 | 182,681 | 0.41x | 10.8% | 15.5% |
| MUL_MAT f16 | 224,511 | 85,823 | 0.38x | 5.5% | 7.3% |
| RMS_NORM_MUL_ROPE | 97,444 | 52,192 | 0.54x | 2.4% | 4.4% |
| RMS_NORM_MUL | 25,379 | 61,128 | 2.41x | 0.6% | 5.2% |
| GLU | 21,867 | 9,020 | 0.41x | 0.5% | 0.8% |
| ADD | 17,389 | 10,885 | 0.63x | 0.4% | 0.9% |
| SET_ROWS | 4,501 | 2,456 | 0.55x | 0.1% | 0.2% |
| MUL_MAT_ADD q6_K | 449 | 311 | 0.69x | 0.0% | 0.0% |
| CPY | 34 | 0 | 0.00x | 0.0% | - |
| GET_ROWS | 19 | 0 | 0.00x | 0.0% | - |
| **Total** | **4,080,230** | **1,175,135** | **0.29x** | | |

### input_length=300: Decode (per token)

| Op | Actual (us) | Estimate (us) | Ratio | Act % | Est % |
|---|---:|---:|---:|---:|---:|
| MUL_MAT q4_K | 43,660 | 27,761 | 0.64x | 33.1% | 35.7% |
| FLASH_ATTN_EXT | 41,440 | 12,838 | 0.31x | 31.4% | 16.5% |
| MUL_MAT q6_K | 17,488 | 8,761 | 0.50x | 13.3% | 11.3% |
| MUL_MAT_ADD q4_K | 14,631 | 8,726 | 0.60x | 11.1% | 11.2% |
| MUL_MAT_ADD q6_K | 7,401 | 4,351 | 0.59x | 5.6% | 5.6% |
| MUL_MAT f16 | 3,376 | 4,177 | 1.24x | 2.6% | 5.4% |
| RMS_NORM_MUL | 1,804 | 3,205 | 1.78x | 1.4% | 4.1% |
| RMS_NORM_MUL_ROPE | 1,528 | 2,670 | 1.75x | 1.2% | 3.4% |
| SET_ROWS | 382 | 2,456 | 6.43x | 0.3% | 3.2% |
| GLU | 201 | 0 | 0.00x | 0.2% | - |
| GET_ROWS | 14 | 0 | 0.00x | 0.0% | - |
| ADD | 11 | 33 | 3.00x | 0.0% | 0.0% |
| CPY | 7 | 0 | 0.0x | 0.0% | - |
| **Total** | **131,943** | **77,816** | **0.59x** | | |

## Raw Per-Op Data (with shapes)

### input_length=18: Prefill Raw (Run 1 / Run 2)

| Op (with shape) | Run 1: count x avg (us) | Run 2: count x avg (us) |
|---|---|---|
| MUL_MAT q4_K m=6144 n=18 k=2048 | 54 x 2,216.3 | 54 x 2,542.7 |
| MUL_MAT q6_K m=2048 n=18 k=6144 | 13 x 5,317.6 | 13 x 5,003.7 |
| MUL_MAT q4_K m=2048 n=18 k=2048 | 56 x 848.0 | 56 x 1,339.9 |
| MUL_MAT q4_K m=2048 n=18 k=6144 | 14 x 3,453.7 | 14 x 2,687.0 |
| MUL_MAT f16 m=1024 n=18 k=2048 | 28 x 1,221.3 | 28 x 1,401.4 |
| FLASH_ATTN_EXT dst(128,16,18,1),  q(128,18,16,1),  k(128,256,8,1),  v(128,256,8,1),  m(256,18,1,1) | 28 x 926.4 | 28 x 997.3 |
| MUL_MAT q4_K m=1024 n=18 k=2048 | 28 x 557.2 | 28 x 935.9 |
| MUL_MAT_VEC q6_K m=151936 n=1 k=2048 | 1 x 10,858.6 | 1 x 17,438.8 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,16,18,1) | 28 x 101.1 | 28 x 127.2 |
| GLU | 28 x 21.5 | 28 x 155.8 |
| RMS_NORM_MUL RMS_NORM(2048,18,1,1) | 55 x 58.5 | 55 x 29.4 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,8,18,1) | 28 x 48.1 | 28 x 53.7 |
| ADD | 55 x 18.8 | 55 x 19.2 |
| MUL_MAT_VEC q4_K m=6144 n=1 k=2048 | 2 x 395.2 | 2 x 411.2 |
| SET_ROWS | 56 x 8.9 | 56 x 9.8 |
| MUL_MAT_ADD MUL_MAT_VEC q6_K m=2048 n=1 k=6144 | 1 x 390.4 | 1 x 416.1 |
| RMS_NORM_MUL RMS_NORM(2048,1,1,1) | 2 x 16.2 | 2 x 22.6 |
| GET_ROWS | 2 x 6.9 | 2 x 7.3 |
| CPY | 1 x 8.2 | 1 x 7.6 |

### input_length=150: Prefill Raw (Run 1 / Run 2)

| Op (with shape) | Run 1: count x avg (us) | Run 2: count x avg (us) |
|---|---|---|
| FLASH_ATTN_EXT dst(128,16,153,1),  q(128,153,16,1),  k(128,256,8,1),  v(128,256,8,1),  m(256,153,1,1) | 28 x 18,235.6 | 28 x 15,932.4 |
| MUL_MAT q4_K m=6144 n=153 k=2048 | 54 x 7,066.0 | 54 x 8,316.5 |
| MUL_MAT q6_K m=2048 n=153 k=6144 | 13 x 18,117.4 | 13 x 16,943.4 |
| MUL_MAT q4_K m=2048 n=153 k=2048 | 56 x 2,987.0 | 56 x 3,407.1 |
| MUL_MAT q4_K m=2048 n=153 k=6144 | 14 x 6,768.1 | 14 x 7,720.9 |
| MUL_MAT f16 m=1024 n=153 k=2048 | 28 x 2,678.2 | 28 x 3,887.3 |
| MUL_MAT q4_K m=1024 n=153 k=2048 | 28 x 1,430.9 | 28 x 1,157.9 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,16,153,1) | 28 x 884.9 | 28 x 1,234.3 |
| MUL_MAT_VEC q6_K m=151936 n=1 k=2048 | 1 x 20,152.8 | 1 x 13,072.9 |
| ADD | 55 x 438.4 | 55 x 90.9 |
| RMS_NORM_MUL RMS_NORM(2048,153,1,1) | 55 x 309.7 | 55 x 216.0 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,8,153,1) | 28 x 438.9 | 28 x 461.9 |
| GLU | 28 x 528.4 | 28 x 285.6 |
| MUL_MAT_VEC q4_K m=6144 n=1 k=2048 | 2 x 453.9 | 2 x 5,639.4 |
| SET_ROWS | 56 x 36.3 | 56 x 39.7 |
| MUL_MAT_ADD MUL_MAT_VEC q6_K m=2048 n=1 k=6144 | 1 x 452.8 | 1 x 472.4 |
| RMS_NORM_MUL RMS_NORM(2048,1,1,1) | 2 x 17.6 | 2 x 31.5 |
| GET_ROWS | 2 x 7.4 | 2 x 7.9 |
| CPY | 1 x 13.4 | 1 x 14.5 |

### input_length=300: Prefill Raw (Run 1 / Run 2)

| Op (with shape) | Run 1: count x avg (us) | Run 2: count x avg (us) |
|---|---|---|
| FLASH_ATTN_EXT dst(128,16,325,1),  q(128,325,16,1),  k(128,512,8,1),  v(128,512,8,1),  m(512,325,1,1) | 28 x 60,148.5 | 28 x 56,563.6 |
| MUL_MAT q4_K m=6144 n=325 k=2048 | 54 x 16,719.4 | 54 x 15,685.3 |
| MUL_MAT q6_K m=2048 n=325 k=6144 | 13 x 33,072.6 | 13 x 32,618.4 |
| MUL_MAT q4_K m=2048 n=325 k=2048 | 56 x 6,401.0 | 56 x 7,017.6 |
| MUL_MAT q4_K m=2048 n=325 k=6144 | 14 x 19,119.5 | 14 x 18,426.3 |
| MUL_MAT f16 m=1024 n=325 k=2048 | 28 x 7,745.4 | 28 x 8,291.1 |
| MUL_MAT q4_K m=1024 n=325 k=2048 | 28 x 3,459.8 | 28 x 3,620.4 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,16,325,1) | 28 x 2,514.5 | 28 x 2,210.4 |
| RMS_NORM_MUL_ROPE RMS_NORM(128,8,325,1) | 28 x 1,012.2 | 28 x 1,223.1 |
| RMS_NORM_MUL RMS_NORM(2048,325,1,1) | 55 x 525.1 | 55 x 396.1 |
| GLU | 28 x 916.5 | 28 x 645.4 |
| ADD | 55 x 432.5 | 55 x 199.9 |
| MUL_MAT_VEC q6_K m=151936 n=1 k=2048 | 1 x 17,245.3 | 1 x 11,112.4 |
| SET_ROWS | 56 x 79.4 | 56 x 81.3 |
| MUL_MAT_VEC q4_K m=6144 n=1 k=2048 | 2 x 483.8 | 2 x 419.1 |
| MUL_MAT_ADD MUL_MAT_VEC q6_K m=2048 n=1 k=6144 | 1 x 494.7 | 1 x 404.1 |
| RMS_NORM_MUL RMS_NORM(2048,1,1,1) | 2 x 24.3 | 2 x 21.5 |
| CPY | 1 x 33.3 | 1 x 34.9 |
| GET_ROWS | 2 x 9.0 | 2 x 9.8 |
