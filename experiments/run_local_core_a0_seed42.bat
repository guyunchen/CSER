@echo off
cd /d E:\SER\CSER
D:\Anaconda3\python.exe experiments\run_experiments.py --config experiments\configs\core_models.yaml --seeds 42 --max-experiments 1 --num-workers 0 --skip-followups > experiments\output\logs\local_core_A0_seed42.current.log 2>&1
