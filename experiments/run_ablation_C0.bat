@echo off
cd /d E:\SER\CSER
D:\Anaconda3\envs\CSER\python.exe experiments\run_experiments.py --config experiments\configs\ablations.yaml --only-ids C0 --num-workers 0 --skip-followups > experiments\output\logs\ablations_C0_launcher.log 2>&1
