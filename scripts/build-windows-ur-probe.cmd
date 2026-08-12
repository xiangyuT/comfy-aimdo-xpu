@echo off
rem Build the Windows Unified Runtime USM interception probe.
rem
rem The probe is deliberately independent of the AIMDO build: it needs only
rem cl.exe and Detours, not the oneAPI DPC++ compiler, so it can be built and
rem run before any decision is made about porting the Linux native allocator
rem hook to Windows.

setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build\ur-probe"
set "OUTPUT_PATH=%BUILD_DIR%\ur_usm_detour_probe.dll"

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
    set "PATH=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.buildtools\bin\%WINDOWS_SDK_VERSION%\x64;%PATH%"
)

if not defined DETOURS_ROOT set "DETOURS_ROOT=%ROOT_DIR%\build\detours-src"
if not defined DETOURS_INCLUDE set "DETOURS_INCLUDE=%DETOURS_ROOT%\include"
if not defined DETOURS_LIB_DIR set "DETOURS_LIB_DIR=%DETOURS_ROOT%\lib.X64"
if not exist "%DETOURS_LIB_DIR%\detours.lib" (
    echo Detours was not found at %DETOURS_LIB_DIR%\detours.lib 1>&2
    echo Run scripts\build-windows-detours.cmd, or set DETOURS_LIB_DIR. 1>&2
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cl.exe /nologo /c /O2 /MD /W3 /I"%DETOURS_INCLUDE%" ^
    "%ROOT_DIR%\tests\ur_usm_detour_probe.c" /Fo"%BUILD_DIR%\ur_usm_detour_probe.obj"
if errorlevel 1 exit /b 1

link.exe /nologo /DLL /OUT:"%OUTPUT_PATH%" "%BUILD_DIR%\ur_usm_detour_probe.obj" ^
    /LIBPATH:"%DETOURS_LIB_DIR%" detours.lib kernel32.lib
if errorlevel 1 exit /b 1

echo built %OUTPUT_PATH%
endlocal
