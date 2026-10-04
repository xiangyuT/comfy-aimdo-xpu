# Windows Intel XPU 编译与 ComfyUI 部署

本文适用于 `dev/xpu-level-zero-vbar` 分支。该分支通过 Intel Level Zero
实现 Windows XPU 的 VBAR 和 PyTorch XPU allocator。

## 编译环境

- Windows 11+
- Visual Studio 2022 Build Tools，安装 Desktop development with C++ 和 Windows SDK
- Intel oneAPI DPC++/C++ Compiler
- Intel 显卡驱动和 Level Zero runtime
- PyTorch XPU 2.8 或更新版本
- Level Zero headers

源码不会携带显卡驱动、oneAPI、Level Zero 二进制或模型文件。

## 编译

在普通 `cmd.exe` 中执行：

```bat
git clone --branch dev/xpu-level-zero-vbar --single-branch ^
  https://github.com/Jianmiao/comfy-aimdo-xpu.git
cd comfy-aimdo-xpu

set ONEAPI_ROOT=F:\oneAPI
set LEVEL_ZERO_INCLUDE=H:\Intel XPU\level-zero-src\include

scripts\build-windows-xpu.cmd
```

`VS_PATH` 未设置时，脚本会通过 `vswhere.exe` 自动定位 Visual Studio。
编译结果为：

```text
comfy_aimdo\aimdo_xpu.dll
```

如果 Level Zero headers 在其他位置，只需调整 `LEVEL_ZERO_INCLUDE`。目录中
应能找到 `ze_api.h` 或 `level_zero\ze_api.h`。

## 安装到 ComfyUI

先完全退出 ComfyUI，然后备份目标目录。将构建结果复制到当前 Python 环境：

```bat
copy /Y comfy_aimdo\aimdo_xpu.dll ^
  "D:\ComfyUI for intel XPU\python_embeded\Lib\site-packages\comfy_aimdo\aimdo_xpu.dll"
```

同时把此分支的 `comfy_aimdo` Python 模块同步到同一个 site-packages 目录；不要只
替换 DLL，因为 Python wrapper 和 native DLL 必须使用同一版本协议。不要覆盖其他
厂商 backend 文件，除非你明确只在 XPU 环境运行。

启动 ComfyUI 时加入：

```text
--enable-dynamic-vram --reserve-vram 2.0
```

日志应出现：

```text
DynamicVRAM support detected and enabled
```

## 验证

在 ComfyUI 的 Python 环境中运行：

```bat
python_embeded\python.exe -c "import torch; from comfy_aimdo import control; print(torch.__version__, torch.xpu.is_available()); print(control.init('xpu')); print(control.init_device(0))"
```

这只验证 DLL、PyTorch XPU 和 Level Zero 初始化。完整工作流仍需单独测试。

## 运行时限制

编译成功不代表每个驱动都支持 VBAR 映射。若日志出现：

```text
cuMemMap(...): Level Zero or SYCL error
VRAM Allocation failed (non OOM)
```

这是运行时的 Level Zero virtual-memory mapping 失败，不是 C/C++ 编译失败，也不
一定是显存不足。可保留 DynamicVRAM，同时让 ComfyUI 使用普通 staged transfers，
或让 VAE 使用 GPU 上的 legacy patcher 绕过 VBAR。更换 oneAPI 版本通常不能替代
兼容的 Intel 显卡驱动。

`aimdo_xpu.dll` 不是按显卡型号单独编译的；它按 Windows、oneAPI、PyTorch ABI
和驱动环境构建。跨机器时建议重新编译并重新验证。
