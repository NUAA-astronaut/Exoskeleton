param([ValidateSet('level','upstairs','downstairs')][string]$Mode='level')
& 'E:\miniconda\envs\pytorch\python.exe' "$PSScriptRoot\run.py" --mode $Mode --viewer
