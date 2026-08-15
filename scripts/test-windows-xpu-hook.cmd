@echo off
rem Build and run the Windows Unified Runtime hook unit test.
rem
rem The test includes src-xpu/ur-usm-detour.c directly and scripts the AIMDO
rem side, so it needs neither a device nor Detours. It is the Windows
rem counterpart of scripts/test-linux-xpu-hook.sh.

setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build\xpu-hook-unit"

if not defined VS_PATH (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS_PATH=%%I"
)
if not defined VS_PATH (
    echo Visual Studio Build Tools were not found. 1>&2
    exit /b 1
)

call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

if not defined WINDOWS_SDK_NUGET set "WINDOWS_SDK_NUGET=%ROOT_DIR%\build\windows-sdk-nuget"
if not defined WINDOWS_SDK_VERSION set "WINDOWS_SDK_VERSION=10.0.26100.0"
if exist "%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt\stddef.h" (
    set "INCLUDE=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\shared;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\um;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\winrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\cppwinrt;%INCLUDE%"
    set "LIB=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\ucrt\x64;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\um\x64;%LIB%"
)

if not defined DETOURS_ROOT set "DETOURS_ROOT=%ROOT_DIR%\build\detours-src"
if not defined DETOURS_INCLUDE set "DETOURS_INCLUDE=%DETOURS_ROOT%\include"
if not defined DETOURS_LIB_DIR set "DETOURS_LIB_DIR=%DETOURS_ROOT%\lib.X64"
if not exist "%DETOURS_LIB_DIR%\detours.lib" (
    echo Detours was not found at %DETOURS_LIB_DIR%\detours.lib 1>&2
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cl.exe /nologo /O2 /MD /W3 /I"%DETOURS_INCLUDE%" ^
    "%ROOT_DIR%\tests\ur_usm_detour_unit.c" ^
    /Fo"%BUILD_DIR%\\" /Fe"%BUILD_DIR%\ur_usm_detour_unit.exe" ^
    /link /LIBPATH:"%DETOURS_LIB_DIR%" detours.lib kernel32.lib
if errorlevel 1 exit /b 1

"%BUILD_DIR%\ur_usm_detour_unit.exe"
if errorlevel 1 exit /b 1

endlocal
