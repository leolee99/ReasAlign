#!/bin/sh

python inference_batch.py --dataset sep_sec --do_judge --judge_model adapters/judge_model --defense_model adapters/defense_model --batch_size 16