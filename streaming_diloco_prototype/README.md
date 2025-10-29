Requirements:

torchx

1. Start lighthouse

`RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000`

2. Start replica groups (see torchft/torchx.py)

`torchx run`
