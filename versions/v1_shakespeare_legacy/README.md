# v1_kafka_legacy

Legacy checkpoints from the first Kafka training run and early transformer experiments.

## Status

**Model code not preserved.** The architecture was modified in-place before versioning was set up. These checkpoints cannot be loaded without manually reconstructing the original architecture.

## Known checkpoints

| File | Notes |
|------|-------|
| `kafka_e980_checkpoint.pt` | e0_b980 from the initial 2-layer Kafka run |
| `transformer_dev_e17_b1350.pt` | End of early transformer training run |
| `transformer_dev_e3_b0.pt` | Early epoch checkpoint |

## Token files

The vocabulary used for these checkpoints is in `tokens/`. It was built from the Kafka + Dostoyevsky corpus at an earlier stage and may differ slightly from v2_kafka_12l tokens.
