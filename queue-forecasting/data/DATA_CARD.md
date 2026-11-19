# Data card: switch queue telemetry

Data collected by Mohamed Ali Kabiri on his emulated network testbed (BMv2 software switches in Mininet).

## What was recorded

- Two software switches in series. **Hop 0 is the bottleneck switch**: its output is limited to 750 packets per second, and its single FIFO queue holds at most 64 packets. **Hop 1 is the downstream switch.**
- Three background UDP flows cross the bottleneck. Each flow switches between ON and OFF periods, and the ON periods are heavy-tailed: most are short and a few are much longer.
- There is one measurement packet every 2 ms, which is 500 per second. At each switch it reports the queue it leaves, as two numbers: the queue depth in packets and the time the measurement packet itself spent in that queue. A receiver records both readings, one row per switch.
- There are 39 runs of 90 s each, with 45,000 measurement packets per run (seq 0 to 44,999). All runs use the same settings, so only the random traffic differs. Runs are numbered in recording order.

## Files

| path | content |
|---|---|
| `runs/run_01.parquet` ... `runs/run_39.parquet` | one file per run, with the columns below |
| `sample.csv` | first 10 s of `run_01` (`t_ms < 10000`), 9,978 rows with the same columns, for a quick look without Parquet |
| `splits.json` | fixed run-wise split: 23 train, 8 validation and 8 test runs |

## Columns

| column | type | unit | meaning |
|---|---|---|---|
| `run_id` | string | | `run_01` ... `run_39` |
| `seq` | int32 | sample | measurement packet number, from 0 to 44,999, one every 2 ms. Lost packets leave gaps. |
| `t_ms` | float64 | ms | receive time, counted from the first received sample of the run. Both rows of a sample have the same value. |
| `hop` | int8 | | 0 = bottleneck switch, 1 = downstream switch |
| `qdepth_pkts` | int16 | packets | depth of the queue the measurement packet leaves at this switch, from 0 to 64 |
| `qdelay_us` | int32 | µs | time the measurement packet spent in that queue |

Each received sample has exactly two rows (hop 0 and hop 1). Rows are sorted by `seq`, then `hop`.

The forecasting target is `qdepth_pkts` at hop 0. Forecast horizons count samples (`seq`), not `t_ms`.

## Known quirks

- **Lost samples (probe loss).** 11,875 of the 1,755,000 measurement packets were lost (0.68%; between 0.22% and 2.59% per run). They show up as missing `seq` values, and nothing is filled in. The losses are not random. For 77% of them, the last bottleneck reading before the loss was 60 packets or more, so most packets are lost when the 64-packet buffer is full. Dropping lost targets therefore removes part of the high end of the target.
- **Rare host stalls.** All switches and hosts run on one machine, and that machine sometimes pauses. In `t_ms` a stall shows up as a jump from one received sample to the next that is well beyond 2 ms per `seq` step. Pauses longer than 20 ms beyond the expected spacing happen 336 times, in 32 runs. Pauses longer than 50 ms happen 9 times, in 6 runs. The longest is 673 ms, in `run_39`. Because `t_ms` is not evenly spaced, use `seq` as the time axis.
- **Very long queue delays.** At 750 packets per second, a full 64-packet buffer drains in about 85 ms. Even so, `qdelay_us` at hop 0 is above 90 ms in 1,423 samples spread over 19 runs (0.08% of samples), and reaches up to 748 ms. Check how your features handle these values.
- **Delay floor.** `qdelay_us` is not zero when the queue is empty. For samples with `qdepth_pkts = 0`, the median per run is between 98 and 125 µs.
- **The downstream queue is mostly empty.** At hop 1, `qdepth_pkts > 0` in between 0.04% and 22% of the samples of a run, and the largest value is 49 packets.

## Row counts and sizes

In total there are 39 files with 3,486,250 rows (1,743,125 received samples × 2 hops), taking 37.7 MB as Parquet with zstd compression. `sample.csv` is 317 KB.

| file | rows | received samples | lost samples | size (KB) |
|---|---:|---:|---:|---:|
| runs/run_01.parquet | 89,202 | 44,601 | 399 | 962 |
| runs/run_02.parquet | 89,336 | 44,668 | 332 | 957 |
| runs/run_03.parquet | 89,562 | 44,781 | 219 | 959 |
| runs/run_04.parquet | 89,178 | 44,589 | 411 | 963 |
| runs/run_05.parquet | 89,674 | 44,837 | 163 | 960 |
| runs/run_06.parquet | 89,236 | 44,618 | 382 | 954 |
| runs/run_07.parquet | 89,522 | 44,761 | 239 | 962 |
| runs/run_08.parquet | 89,498 | 44,749 | 251 | 957 |
| runs/run_09.parquet | 89,388 | 44,694 | 306 | 984 |
| runs/run_10.parquet | 89,162 | 44,581 | 419 | 969 |
| runs/run_11.parquet | 89,252 | 44,626 | 374 | 963 |
| runs/run_12.parquet | 89,658 | 44,829 | 171 | 980 |
| runs/run_13.parquet | 89,444 | 44,722 | 278 | 979 |
| runs/run_14.parquet | 89,220 | 44,610 | 390 | 979 |
| runs/run_15.parquet | 89,546 | 44,773 | 227 | 969 |
| runs/run_16.parquet | 89,340 | 44,670 | 330 | 985 |
| runs/run_17.parquet | 89,364 | 44,682 | 318 | 972 |
| runs/run_18.parquet | 89,606 | 44,803 | 197 | 964 |
| runs/run_19.parquet | 89,732 | 44,866 | 134 | 961 |
| runs/run_20.parquet | 89,442 | 44,721 | 279 | 967 |
| runs/run_21.parquet | 89,530 | 44,765 | 235 | 957 |
| runs/run_22.parquet | 89,356 | 44,678 | 322 | 969 |
| runs/run_23.parquet | 89,584 | 44,792 | 208 | 967 |
| runs/run_24.parquet | 89,140 | 44,570 | 430 | 976 |
| runs/run_25.parquet | 89,566 | 44,783 | 217 | 957 |
| runs/run_26.parquet | 89,806 | 44,903 | 97 | 957 |
| runs/run_27.parquet | 89,286 | 44,643 | 357 | 955 |
| runs/run_28.parquet | 89,446 | 44,723 | 277 | 958 |
| runs/run_29.parquet | 89,730 | 44,865 | 135 | 958 |
| runs/run_30.parquet | 89,552 | 44,776 | 224 | 955 |
| runs/run_31.parquet | 89,576 | 44,788 | 212 | 957 |
| runs/run_32.parquet | 89,404 | 44,702 | 298 | 969 |
| runs/run_33.parquet | 89,474 | 44,737 | 263 | 962 |
| runs/run_34.parquet | 89,472 | 44,736 | 264 | 956 |
| runs/run_35.parquet | 89,116 | 44,558 | 442 | 998 |
| runs/run_36.parquet | 87,670 | 43,835 | 1,165 | 1,011 |
| runs/run_37.parquet | 89,676 | 44,838 | 162 | 980 |
| runs/run_38.parquet | 89,404 | 44,702 | 298 | 972 |
| runs/run_39.parquet | 89,100 | 44,550 | 450 | 961 |

## Splits

`splits.json` lists the run ids for train (23), validation (8) and test (8). It was made in four steps:

1. Sort the 39 run ids.
2. Permute them with `numpy.random.default_rng(0).permutation(39)`.
3. Take the first 23 for train, the next 8 for validation and the last 8 for test.
4. Sort each list.

A run is never split between sets, so no recording appears in two of them. Fit models on train. Choose between them on validation, or with cross-validation grouped by run on the training runs. Use the test runs once, at the end.
