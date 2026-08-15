@echo off
rem Configure the Intel oneAPI DPC++ compiler environment.
rem
rem The oneAPI setvars.bat in the recorded build environment fails to invoke
rem any component vars.bat ("'vars.bat' is not recognized"), so nothing lands
rem on PATH/INCLUDE/LIB. This script sets the compiler variables directly from
rem the installation layout, and still prefers setvars.bat when it works.
rem
rem Call it with "call", not directly: it exports variables to the caller.

if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"

if defined VS_PATH if not defined VS2022INSTALLDIR set "VS2022INSTALLDIR=%VS_PATH%"

if not defined ONEAPI_COMPILER_ROOT (
    for /f "delims=" %%I in ('dir /b /ad /o-n "%ONEAPI_ROOT%\compiler" 2^>nul') do (
        if not defined ONEAPI_COMPILER_ROOT set "ONEAPI_COMPILER_ROOT=%ONEAPI_ROOT%\compiler\%%I"
    )
)
if not defined ONEAPI_COMPILER_ROOT (
    echo Intel oneAPI DPC++ compiler was not found below "%ONEAPI_ROOT%\compiler". 1>&2
    exit /b 1
)
if not exist "%ONEAPI_COMPILER_ROOT%\bin\icx-cl.exe" (
    echo icx-cl.exe was not found in "%ONEAPI_COMPILER_ROOT%\bin". 1>&2
    exit /b 1
)

set "PATH=%ONEAPI_COMPILER_ROOT%\bin;%PATH%"
set "INCLUDE=%ONEAPI_COMPILER_ROOT%\include;%ONEAPI_COMPILER_ROOT%\include\sycl;%INCLUDE%"
set "LIB=%ONEAPI_COMPILER_ROOT%\lib;%ONEAPI_COMPILER_ROOT%\opt\compiler\lib;%LIB%"

where icx-cl >nul 2>&1
if errorlevel 1 (
    echo icx-cl.exe is still not on PATH after configuring oneAPI. 1>&2
    exit /b 1
)

echo using oneAPI compiler at %ONEAPI_COMPILER_ROOT%
exit /b 0
