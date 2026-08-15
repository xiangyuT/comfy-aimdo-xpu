@echo off
rem Build Microsoft Detours against the project-local NuGet Windows SDK.
rem The Visual Studio Build Tools installation used for this project does not
rem include a Windows SDK, so Detours' own nmake cannot find windows.h without
rem the same INCLUDE/LIB setup that scripts\build-windows-xpu.cmd performs.
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "DETOURS_DIR=%ROOT_DIR%\build\detours-src"

if not exist "%DETOURS_DIR%\src\detours.cpp" (
    echo Detours source was not found at %DETOURS_DIR%. 1>&2
    echo Run: git clone --depth 1 https://github.com/microsoft/Detours.git "%DETOURS_DIR%" 1>&2
    exit /b 1
)

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

pushd "%DETOURS_DIR%\src"
set "DETOURS_TARGET_PROCESSOR=X64"
rem Build only the library. The top-level makefile also builds samples, which
rem require the .NET strong-name tool and are not used by AIMDO.
nmake
set "NMAKE_STATUS=%ERRORLEVEL%"
popd
if not "%NMAKE_STATUS%"=="0" exit /b %NMAKE_STATUS%

if not exist "%DETOURS_DIR%\lib.X64\detours.lib" (
    echo Detours build did not produce lib.X64\detours.lib 1>&2
    exit /b 1
)

echo built %DETOURS_DIR%\lib.X64\detours.lib
endlocal
