@echo off
rem lungmap for Windows.
rem   Double-click this file, or drag DICOM files or a folder of them onto it.
rem   The first run downloads Python, the libraries and the model (about 1.5 GB,
rem   several minutes). Later runs start in seconds.
rem   Results go in a "lungmap_output" folder next to each scan, which opens when done.

setlocal
chcp 65001 >nul
title lungmap
pushd "%~dp0"

rem Python environment and model live in %LOCALAPPDATA%\lungmap, not in this
rem folder: keeps ~1 GB of libraries out of OneDrive-synced Desktop/Documents,
rem and survives replacing this folder with a newer download.
set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\lungmap\venv"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
    echo First run: installing uv, which sets up Python for lungmap ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>nul
    if errorlevel 1 goto :failed
)

set "INPUTS=%*"
if "%~1"=="" (
    echo.
    echo Drag a DICOM file or a folder into this window, then press Enter:
    set /p "INPUTS=> "
)
if not defined INPUTS (
    echo Nothing to do.
    goto :end
)

echo.
uv run --locked lungmap --open %INPUTS%
if errorlevel 1 goto :failed
echo.
echo Done.
goto :end

:failed
echo.
echo Something went wrong - see the messages above.

:end
popd
if not defined CI pause
