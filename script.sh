#!/bin/bash
nohup /home/liubo/.conda/envs/torch/bin/python quant_train.py > script.log 2>&1 &


# for PowerShell error like this:
# 无法加载文件 E:\Documents_E\vscode\python\I-ViT.venv\Scripts\Activate.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 about_Execution_Policies。
# 
# Solution:
# Open PowerShell and run:
#       Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#
# Note that this only sets the execution policy for the current PowerShell session.