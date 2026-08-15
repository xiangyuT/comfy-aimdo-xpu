@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "ROOT_DIR=%%~fI"
set "BUILD_DIR=%ROOT_DIR%\build\xpu-windows"
set "OUTPUT_PATH=%ROOT_DIR%\comfy_aimdo\aimdo_xpu.dll"

if not defined VS_PATH (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VS_PATH=%%I"
)
if not defined VS_PATH (
    echo Visual Studio Build Tools were not found. 1>&2
    exit /b 1
)

call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1

if not defined ONEAPI_ROOT set "ONEAPI_ROOT=%ProgramFiles(x86)%\Intel\oneAPI"
if not exist "%ONEAPI_ROOT%\setvars.bat" (
    echo Intel oneAPI setvars.bat was not found below ONEAPI_ROOT. 1>&2
    exit /b 1
)
call "%ONEAPI_ROOT%\setvars.bat" --force
where icx-cl >nul 2>&1
if errorlevel 1 (
    rem Some oneAPI installations fail to run their component vars.bat files
    rem from setvars.bat, which leaves the compiler off PATH entirely.
    call "%~dp0setup-oneapi-env.cmd"
    if errorlevel 1 exit /b 1
)

if not defined WINDOWS_SDK_NUGET set "WINDOWS_SDK_NUGET=%ROOT_DIR%\build\windows-sdk-nuget"
if not defined WINDOWS_SDK_VERSION set "WINDOWS_SDK_VERSION=10.0.26100.0"
if exist "%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt\stddef.h" (
    set "SDK_CPP=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c"
    set "SDK_X64=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c"
    set "SDK_TOOLS=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.buildtools"
    set "INCLUDE=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\ucrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\shared;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\um;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\winrt;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp\c\Include\%WINDOWS_SDK_VERSION%\cppwinrt;%INCLUDE%"
    set "LIB=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\ucrt\x64;%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.cpp.x64\c\um\x64;%LIB%"
    set "PATH=%WINDOWS_SDK_NUGET%\microsoft.windows.sdk.buildtools\bin\%WINDOWS_SDK_VERSION%\x64;%PATH%"
)

if not defined LEVEL_ZERO_INCLUDE set "LEVEL_ZERO_INCLUDE=%ROOT_DIR%\build\level-zero-src\include"
if not exist "%LEVEL_ZERO_INCLUDE%\ze_api.h" if not exist "%LEVEL_ZERO_INCLUDE%\level_zero\ze_api.h" (
    echo Level Zero headers were not found below LEVEL_ZERO_INCLUDE. 1>&2
    echo Clone https://github.com/oneapi-src/level-zero into build\level-zero-src or set LEVEL_ZERO_INCLUDE. 1>&2
    exit /b 1
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
del /q "%BUILD_DIR%\*.obj" "%BUILD_DIR%\ze_loader.*" 2>nul

lib.exe /nologo /def:"%ROOT_DIR%\src-xpu\ze_loader.def" /machine:x64 /out:"%BUILD_DIR%\ze_loader.lib"
if errorlevel 1 exit /b 1

set "COMMON_FLAGS=/nologo /c /O2 /MD /DAIMDO_XPU /I"%ROOT_DIR%\src" /I"%ROOT_DIR%\src-win" /FIcompiler.h"
for %%S in (
    control.c
    debug.c
    hostbuf-decommit.c
    hostbuf-file-reader.c
    hostbuf-prewarm.c
    hostbuf.c
    model-vbar.c
    pyt-cu-plug-alloc.c
    pyt-cu-plug-alloc-async.c
    vrambuf.c
    xfer-file.c
) do (
    cl.exe %COMMON_FLAGS% "%ROOT_DIR%\src\%%S" /Fo"%BUILD_DIR%\%%~nS.obj"
    if errorlevel 1 exit /b 1
)

for %%S in (hostbuf-plat.c model-mmap.c thread-plat.c xfer-file-plat.c shmem-detect.c) do (
    cl.exe %COMMON_FLAGS% "%ROOT_DIR%\src-win\%%S" /Fo"%BUILD_DIR%\win-%%~nS.obj"
    if errorlevel 1 exit /b 1
)

cl.exe %COMMON_FLAGS% "%ROOT_DIR%\src-xpu\stubs.c" /Fo"%BUILD_DIR%\xpu-stubs.obj"
if errorlevel 1 exit /b 1
cl.exe %COMMON_FLAGS% /I"%LEVEL_ZERO_INCLUDE%" /I"%DETOURS_INCLUDE%" ^
    "%ROOT_DIR%\src-xpu\ze-detour.c" /Fo"%BUILD_DIR%\xpu-ze-detour.obj"
if errorlevel 1 exit /b 1
cl.exe %COMMON_FLAGS% /I"%DETOURS_INCLUDE%" ^
    "%ROOT_DIR%\src-xpu\ur-usm-detour.c" /Fo"%BUILD_DIR%\xpu-ur-usm-detour.obj"
if errorlevel 1 exit /b 1
cl.exe %COMMON_FLAGS% /EHsc /I"%LEVEL_ZERO_INCLUDE%" ^
    "%ROOT_DIR%\src-xpu\ze-tracer.cpp" /Fo"%BUILD_DIR%\xpu-ze-tracer.obj"
if errorlevel 1 exit /b 1
icx-cl.exe /nologo /c /O2 /MD /EHsc /std:c++17 -fsycl /DAIMDO_XPU ^
    /I"%ROOT_DIR%\src" /I"%LEVEL_ZERO_INCLUDE%" ^
    "%ROOT_DIR%\src-xpu\dispatch.cpp" /Fo"%BUILD_DIR%\xpu-dispatch.obj"
if errorlevel 1 exit /b 1
icx-cl.exe /nologo -fsycl /LD /Fe:"%OUTPUT_PATH%" ^
    "%BUILD_DIR%\control.obj" ^
    "%BUILD_DIR%\debug.obj" ^
    "%BUILD_DIR%\hostbuf-decommit.obj" ^
    "%BUILD_DIR%\hostbuf-file-reader.obj" ^
    "%BUILD_DIR%\hostbuf-prewarm.obj" ^
    "%BUILD_DIR%\hostbuf.obj" ^
    "%BUILD_DIR%\model-vbar.obj" ^
    "%BUILD_DIR%\pyt-cu-plug-alloc.obj" ^
    "%BUILD_DIR%\pyt-cu-plug-alloc-async.obj" ^
    "%BUILD_DIR%\vrambuf.obj" ^
    "%BUILD_DIR%\xfer-file.obj" ^
    "%BUILD_DIR%\win-hostbuf-plat.obj" ^
    "%BUILD_DIR%\win-model-mmap.obj" ^
    "%BUILD_DIR%\win-thread-plat.obj" ^
    "%BUILD_DIR%\win-xfer-file-plat.obj" ^
    "%BUILD_DIR%\win-shmem-detect.obj" ^
    "%BUILD_DIR%\xpu-stubs.obj" ^
    "%BUILD_DIR%\xpu-ze-detour.obj" ^
    "%BUILD_DIR%\xpu-ur-usm-detour.obj" ^
    "%BUILD_DIR%\xpu-ze-tracer.obj" ^
    "%BUILD_DIR%\xpu-dispatch.obj" ^
    /link /LIBPATH:"%BUILD_DIR%" /LIBPATH:"%DETOURS_LIB_DIR%" ^
    ze_loader.lib detours.lib dxgi.lib dxguid.lib onecore.lib
if errorlevel 1 exit /b 1

echo built %OUTPUT_PATH%
endlocal
