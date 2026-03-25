@echo off
setlocal enabledelayedexpansion

:: Set the number of times to run the script
set ITERATIONS=10

:: Loop through the specified number of iterations
for /L %%i in (1,1,%ITERATIONS%) do (
    echo Running iteration %%i of %ITERATIONS%
    python .\selfdeblur_levin.py --save_path results/levinIllPosedNoSeed/Run%%i
    echo Iteration %%i completed
    echo ----------------------------
)

echo All iterations completed!
pause
