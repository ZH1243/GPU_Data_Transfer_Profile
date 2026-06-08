#!/usr/bin/env bash
set -euo pipefail

for i in {1..10}; do
  nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    -o "a2a_separate_${i}m" \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=8 nvlink_all_to_all_copy_engine_test.py \
    --nbytes "${i}M" --copy-mode separate --iters 100 --check
done


for i in {8,32,128,256,512,1024,2048}; do
  nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    -o "./nvlink_batch_address_layout_test/src_discontiuous/4*${i}m_batch" \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    --force-overwrite=true \
    torchrun --standalone --nproc_per_node=8 nvlink_batch_address_layout_test.py \
      --copy-size "${i}K" --copies-per-iter 4 --layout src-discontinuous \
      --gap-size 10M --copy-mode batch --check
done

