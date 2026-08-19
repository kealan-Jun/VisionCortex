# VisionCortex RTX 4090 deployment

This directory is the production installer entry point for a Windows RTX 4090 node.
It preserves the validated CV/evidence rules while replacing 4060 hardware limits with
a 24 GB, all-CUDA, dynamic-view execution profile.

## Install

1. Extract the complete package to a short English path, for example
   `D:\VisionCortex-RTX4090-Complete-<commit>`.
2. Install a current NVIDIA RTX 4090 driver. The package does not redistribute a driver.
3. Open PowerShell in the extracted directory and run:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\deployment\rtx4090\01-Install-And-Validate.ps1
   ```

The installer verifies every packaged byte, installs bundled Python 3.12 and all Python
dependencies without using the internet, checks both 21-class YOLO weights, prompts for
the Ark key without echoing it, and builds TensorRT engines on the RTX 4090 itself.

## Run

```powershell
.\deployment\rtx4090\02-Start-Web.ps1
```

Stop the service with `03-Stop-Web.ps1`. The Web service may start without NAS access;
NAS collection discovery will become available when the configured index and archive
share are connected. Input view count and role composition come from the selected index
record and are never fixed to six cameras.

## Storage

- Local input staging: `D:\VisionCortex4090\Input`
- Runtime and logs: `D:\VisionCortex4090\Runtime`
- Reusable implementation cache: `D:\VisionCortex4090\Cache`
- Original input and accepted output: configured NAS index/archive locations

The installer never copies original experiment video into the source tree. TensorRT
`.engine` files and `.venv` are deliberately generated on the target machine and are not
portable package assets.

