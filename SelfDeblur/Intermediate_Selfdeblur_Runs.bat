@echo off
setlocal enabledelayedexpansion

:: First set of runs (3000 iterations)
echo Running first set with 3000 iterations (default)
echo ----------------------------

:: First run - normal run
echo Running iteration 1 of 3 (normal run)
python .\selfdeblur_levin.py --save_path results/levinIntermResults/200/Run1
echo Iteration 1 completed
echo ----------------------------

:: Second run - using intermediate results from first run
echo Running iteration 2 of 3 (using intermediate results)
python .\selfdeblur_levin.py --data_path results/levinIntermResults/200/Run1 --Interm --Interm_path img_x.png --save_path results/levinIntermResults/200/Run2
echo Iteration 2 completed
echo ----------------------------

:: Third run - using intermediate results from second run
echo Running iteration 3 of 3 (using intermediate results)
python .\selfdeblur_levin.py --data_path results/levinIntermResults/200/Run2 --Interm --Interm_path img_x_x.png --save_path results/levinIntermResults/200/Run3
echo Iteration 3 completed
echo ----------------------------

echo All iterations completed!
pause
