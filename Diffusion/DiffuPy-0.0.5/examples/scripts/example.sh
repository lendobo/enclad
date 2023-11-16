#!/usr/bin/env bash

# network=network
# data=data file
# output=output file results
# method=diffusion method (i.e., gm, ml, raw and z)
# binarize=binarize labels (default True)
# absolute_value=apply absolute value to logFC in data (default True)
# threshold=threshold to apply if logFC in data
# p_value=statistical significance (default 0.05)



python3 -m diffupath diffusion diffuse \
  --network ../networks/sample_network_2.csv \
  --data ../datasets/sample_dataset.csv \
  --method raw \

