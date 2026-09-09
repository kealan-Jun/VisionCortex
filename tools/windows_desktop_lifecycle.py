"""An application-owned Windows process tree, tied to its desktop parent's life."""
from __future__ import annotations

import ctypes
import os
import sys
import threading


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", ctypes.c_uint32), ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t), ("active_process_limit", ctypes.c_uint32),
                ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32), ("scheduling", ctypes.c_uint32)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


_OWNED_JOB = None


def supervise_desktop_parent(parent_pid: int) -> None:
    """No process lookup/reuse: hold the original parent HANDLE until it exits."""
    global _OWNED_JOB
    if sys.platform != "win32" or parent_pid <= 0:
        raise RuntimeError("Desktop process supervision requires a Windows parent")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, args, result in (
        ("CreateJobObjectW", [ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
        ("SetInformationJobObject", [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int),
        ("AssignProcessToJobObject", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ("GetCurrentProcess", [], ctypes.c_void_p),
        ("OpenProcess", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
        ("WaitForSingleObject", [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
        ("CloseHandle", [ctypes.c_void_p], ctypes.c_int),
        ("GetStdHandle", [ctypes.c_int32], ctypes.c_void_p),
        ("GetFileType", [ctypes.c_void_p], ctypes.c_uint32),
        ("ReadFile", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                      ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p], ctypes.c_int),
        ("PeekNamedPipe", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p], ctypes.c_int),
    ):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
    command_pipe = kernel.GetStdHandle(-10)
    if not command_pipe or kernel.GetFileType(command_pipe) != 3:
        raise RuntimeError("Desktop process supervision requires its command pipe")
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "Cannot create the desktop process job")
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise OSError(error, "Cannot set desktop process ownership")
    parent = kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE only
    if not parent:
        kernel.CloseHandle(job)
        raise RuntimeError("The desktop application is no longer running")
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(parent)
        kernel.CloseHandle(job)
        raise OSError(error, "Cannot supervise the application process tree")
    _OWNED_JOB = job  # Not inheritable: process exit closes it and all descendants.

    def watch_parent():
        while True:
            result = kernel.WaitForSingleObject(parent, 500)
            if result != 0x102:  # Parent exited, or wait failed: fail closed.
                os._exit(0 if result == 0 else 1)

    def watch_commands():
        # Blocking stdin reads can interfere with native library initialization
        # on Windows. Read only bytes already available in the owned pipe.
        buffer = ctypes.create_string_buffer(4096)
        received, available = ctypes.c_uint32(), ctypes.c_uint32()
        idle = threading.Event()
        pending = b""
        while kernel.PeekNamedPipe(command_pipe, None, 0, None, ctypes.byref(available), None):
            if not available.value:
                idle.wait(0.1)
                continue
            if not kernel.ReadFile(command_pipe, buffer, min(available.value, len(buffer)),
                                   ctypes.byref(received), None):
                break
            if not received.value:
                os._exit(0)
            lines = (pending + buffer.raw[:received.value]).split(b"\n")
            pending = lines.pop()
            if any(line.strip() == b"stop" for line in lines):
                os._exit(0)
            if len(pending) > 4096:
                os._exit(1)
        os._exit(0 if ctypes.get_last_error() == 109 else 1)

    threading.Thread(target=watch_parent, daemon=True).start()
    threading.Thread(target=watch_commands, daemon=True).start()
