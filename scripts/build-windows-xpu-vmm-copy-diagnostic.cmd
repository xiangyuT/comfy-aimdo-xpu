@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build\xpu-windows"
set "OUTPUT_PATH=%BUILD_DIR%\diagnose-windows-xpu-vmm-copy.exe"

if not defined VS_PATH (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS_PATH=%%I"
)
if not defined VS_PATH exit /b 1
call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
call "%ONEAPI_ROOT%\setvars.bat" --force
where icx-cl >nul 2>&1
if errorlevel 1 (
    call "%~dp0setup-oneapi-env.cmd"
    if errorlevel 1 exit /b 1
)

if not defined WINDOWS_SDK_NUGET set "WINDOWS_SDK_NUGET=%ROOT_DIR%\build\windows-sdk-nuget"
if not defined WINDOWS_SDK_VERSION set "WINDOWS_SDK_VERSION=10.0.26100.0"
if exist "%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt\stddef.h" (
    set "INCLUDE=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\shared;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\um;%INCLUDE%"
    set "LIB=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\ucrt\x64;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\um\x64;%LIB%"
    set "PATH=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.buildtools\bin\%WINDOWS_SDK_VERSION%\x64;%PATH%"
)

if not defined LEVEL_ZERO_INCLUDE set "LEVEL_ZERO_INCLUDE=%ROOT_DIR%\build\level-zero-src\include"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
lib.exe /nologo /def:"%ROOT_DIR%\tests\ze_loader_diagnostic.def" /machine:x64 /out:"%BUILD_DIR%\ze_loader.lib"
if errorlevel 1 exit /b 1

icx-cl.exe /nologo /EHsc /std:c++17 /O2 /MD -fsycl ^
    /I"%LEVEL_ZERO_INCLUDE%" ^
    "%ROOT_DIR%\tests\diagnose_windows_xpu_vmm_copy.cpp" ^
    /Fe:"%OUTPUT_PATH%" /link /LIBPATH:"%BUILD_DIR%" ze_loader.lib
if errorlevel 1 exit /b 1

echo built %OUTPUT_PATH%
endlocal
