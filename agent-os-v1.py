#!/usr/bin/env python3
"""Cyber Agent: Multi-Agent System with LLMs, SEO, GUI, N64, Ghidra, Archive, SSH, Cydia.
- Tools: Grouped by category (SEO, GUI, Memory, etc.) with comments.
- Agent Logic: Reasoning, parsing, chat/GUI modes at the end.
This structure allows easy editing: If Gemini (Llama.cpp) has issues, edit only that section.
For speed: Gemini now uses a more robust n_gpu_layers setting for Apple Metal.
"""
import os
import sys
import json
import subprocess
import pathlib  # Imported pathlib
import datetime
import re
import shlex
import ast
import importlib.util
import traceback
import threading
import time
import tempfile
import shutil
import os
import sys
import json
import subprocess
import pathlib  # Imported pathlib
import datetime
import re
import shlex
import ast
import importlib.util
import traceback
import threading
import time
import tempfile
import shutil
import struct  # Import struct for Ghidra agent
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
import struct  # Import struct for Ghidra agent
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
import os
import sys
import json
import subprocess
import pathlib # Imported pathlib
import datetime
import re
import shlex
import ast
import importlib.util
import traceback
import threading
import shutil
import struct
import tempfile
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import time
import urllib.request
import tarfile
import stat
import os
import sys
import json
import subprocess
import pathlib  # Imported pathlib
import datetime
import re
import shlex
import ast
import importlib.util
import traceback
import threading
import time
import tempfile
import shutil
import struct  # Import struct for Ghidra agent
from typing import Dict, Any, List, Union, Optional, Tuple, Callable


































# --- Utility Function for Backend Setup ---
def command_exists(cmd: str) -> bool:
    """Check if a command exists in the system PATH."""
    return subprocess.run(['which', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

# --- Hardcoded paths ---
# Consider making these configurable via environment variables or a config file
SCRIPT_PATH = "/Users/rosty/Desktop/Anacharssis_Z2.py"
GABRIEL_PATH = "/Users/rosty/Desktop/Gabriel_Z1.py"
DATA_DIR = "/Users/rosty/Desktop/AI/Data"
MEMORY_DIR = "/Users/rosty/Desktop/AI/Memory"
ACTIVE_MEMORY_DIR = "/Users/rosty/Desktop/AI/Agents/Active_Memory"
COMFY_JSON_PATH = "/Users/rosty/Desktop/AI/Images/comfy_json/comfy.json"
GEMINI_PATH = "/Users/rosty/Desktop/gemini1.5pro.gguf"
DEEPSEEK_MODEL = "deepseek-r1:7b"
QWEN_PATH = ""  # Set this to the path of your downloaded Qwen model directory

# --- Ghidra Paths ---
GHIDRA_PATH = "/Applications/ghidra_11.0.1_PUBLIC"  # Adjust this path to your Ghidra installation
GHIDRA_HEADLESS_PATH = os.path.join(GHIDRA_PATH, "support", "analyzeHeadless")
GHIDRA_OUTPUT_DIR = "/Users/rosty/Desktop/Ghidra_Output"  # Default output directory for Ghidra results
os.makedirs(GHIDRA_OUTPUT_DIR, exist_ok=True)  # Ensure output dir exists

# --- Cydia Module Path ---
CYDIA_MODULES_PATH = "/Users/rosty/Desktop/cydia"

# --- Global variables for LLM ---
BACKEND = None
MODEL = None
MODEL_PATH = None
MODEL_DIR = None

# --- Global for Dynamic Tools from Cydia ---
# This dictionary will hold tools loaded from Cydia Python modules
CYDIA_DYNAMIC_TOOLS: Dict[str, Callable] = {}

# --- Tool Definitions ---
# Archive & SSH Tools
def archive(**kwargs):
    try:
        directory = kwargs.get("directory", ".")
        output = kwargs.get("output", "archive.tar.gz")
        cmd = ["tar", "-czf", output, "-C", directory, "."]
        subprocess.check_call(cmd)
        return f"Archived '{directory}' to '{output}'."
    except Exception as e:
        return f"Error archiving: {e}"

def extract(**kwargs):
    try:
        archive_file = kwargs.get("archive", "archive.tar.gz")
        output_dir = kwargs.get("output", ".")
        os.makedirs(output_dir, exist_ok=True)
        cmd = ["tar", "-xzf", archive_file, "-C", output_dir]
        subprocess.check_call(cmd)
        return f"Extracted '{archive_file}' to '{output_dir}'."
    except Exception as e:
        return f"Error extracting: {e}"

def list_contents(**kwargs):
    try:
        archive_file = kwargs.get("archive", "archive.tar.gz")
        cmd = ["tar", "-tzf", archive_file]
        result = subprocess.check_output(cmd, text=True)
        return result
    except Exception as e:
        return f"Error listing contents: {e}"

def convert(**kwargs):
    try:
        input_file = kwargs.get("input")
        output_format = kwargs.get("output_format", "json")
        if not input_file:
            return "Error: Missing 'input' argument for convert."
        # Placeholder for actual conversion logic
        return f"Converted '{input_file}' to '{output_format}'."
    except Exception as e:
        return f"Error converting: {e}"

def manifest(**kwargs):
    try:
        directory = kwargs.get("directory", ".")
        manifest_file = kwargs.get("output", "MANIFEST.txt")
        with open(manifest_file, 'w') as f:
            for root, dirs, files in os.walk(directory):
                level = root.replace(directory, '').count(os.sep)
                indent = ' ' * 2 * level
                f.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    f.write(f"{subindent}{file}\n")
        return f"Manifest created: '{manifest_file}' for directory '{directory}'."
    except Exception as e:
        return f"Error creating manifest: {e}"

def verify(**kwargs):
    try:
        archive_file = kwargs.get("archive", "archive.tar.gz")
        cmd = ["tar", "-tzf", archive_file]
        subprocess.check_call(cmd)
        return f"Archive '{archive_file}' verified successfully."
    except Exception as e:
        return f"Error verifying archive: {e}"

def check_and_fix(**kwargs):
    try:
        directory = kwargs.get("directory", ".")
        issues_found = []
        fixed_issues = []
        # Check for common issues (placeholder logic)
        if not os.path.exists(os.path.join(directory, ".git")):
            issues_found.append("No .git directory found.")
        # Fix issues (placeholder)
        if issues_found:
            fixed_issues.append("Performed generic fix for found issues.")
        if fixed_issues:
            return f"Issues found and fixed in '{directory}': {', '.join(fixed_issues)}"
        else:
            return f"No issues found in '{directory}'."
    except Exception as e:
        return f"Error checking/fixing: {e}"

def append(**kwargs):
    try:
        file_path = kwargs.get("file")
        content = kwargs.get("content")
        if not file_path or content is None:
            return "Error: Missing 'file' or 'content' argument for append."
        with open(file_path, 'a') as f:
            f.write(content + '\n')
        return f"Appended content to '{file_path}'."
    except Exception as e:
        return f"Error appending: {e}"

def patch(**kwargs):
    try:
        file_path = kwargs.get("file")
        patch_data = kwargs.get("patch") # Expecting a dict {line_number: new_content}
        if not file_path or not patch_data:
            return "Error: Missing 'file' or 'patch' argument for patch."
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line_num_str, new_content in patch_data.items():
            try:
                line_num = int(line_num_str)
                if 1 <= line_num <= len(lines):
                    lines[line_num - 1] = new_content + '\n' # Line numbers are usually 1-based
                else:
                    return f"Error: Line number {line_num} out of range for file '{file_path}'."
            except ValueError:
                return f"Error: Invalid line number '{line_num_str}' in patch data."
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return f"Patched file '{file_path}'."
    except Exception as e:
        return f"Error patching: {e}"

def do_shelly(**kwargs):
    try:
        ip = kwargs.get("ip")
        command = kwargs.get("command")
        if not ip or not command:
            return "Error: Missing 'ip' or 'command' argument for do_shelly."
        full_cmd = f"ssh root@{ip} '{command}'"
        result = subprocess.check_output(full_cmd, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return f"SSH command failed: {e}"
    except Exception as e:
        return f"Error executing SSH command: {e}"

def zharko_tut(**kwargs):
    # Placeholder for zharko_tut functionality
    return "Zharko tutorial placeholder executed."

def extract_data(**kwargs):
    # Placeholder for extract_data functionality
    return "Extract data placeholder executed."

# SEO Tool
def seo_analyze_site(args: Union[Dict[str, str], str]) -> str:
    """Placeholder for SEO analysis."""
    url = args.get('url') if isinstance(args, dict) else args
    if not url:
        return "Error: No URL provided for SEO analysis."
    # Placeholder logic for SEO analysis
    return f"SEO analysis for {url}: Domain age: Unknown, Backlinks: 0, Keywords: ['placeholder']"

# GUI Tool
def recognize_gui_elements(description: str) -> str:
    """Placeholder for GUI element recognition."""
    # Placeholder logic for GUI recognition
    return f"Recognized GUI elements matching description: {description}"

# Shell Tool
def run_shell_tool(command: str) -> str:
    """Run a shell command and return the output."""
    try:
        result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return result
    except subprocess.CalledProcessError as e:
        return f"Shell command failed with exit code {e.returncode}: {e.output}"
    except Exception as e:
        return f"Error running shell command: {e}"

# Math & Data Processing Tools
class Calculator:
    @staticmethod
    def add(a, b): return a + b
    @staticmethod
    def subtract(a, b): return a - b
    @staticmethod
    def multiply(a, b): return a * b
    @staticmethod
    def divide(a, b):
        if b == 0: return "Error: Division by zero"
        return a / b

def train_screen_model(**kwargs):
    # Placeholder for training screen model
    return "Screen model training placeholder executed."

# Image & Screenshot Tools
def generate_comfy_image(**kwargs):
    # Placeholder for ComfyUI image generation
    return "ComfyUI image generation placeholder executed."

def take_screenshot(**kwargs):
    # Placeholder for taking a screenshot
    return "Screenshot taken placeholder executed."

# Code Analysis & Execution Tools
def read_and_analyze(**kwargs):
    # Placeholder for reading and analyzing code
    return "Read and analyze placeholder executed."

def execute_function(filename: str, func_name: str, *args) -> str:
    """Execute a function from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("module.name", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, func_name)
        result = func(*args)
        return str(result)
    except Exception as e:
        return f"Error executing function: {e}"

# Source Editing Tool
def edit_source_with_nano(args: Union[Dict[str, str], str]) -> str:
    """Edit the main script source using nano and restart."""
    # Expect args to be a dictionary with 'changes' key
    if isinstance(args, str):
        # Fallback if LLM provides a string (try to parse as JSON)
        try:
            args = json.loads(args)
        except:
            return "Error: Invalid arguments for edit_source_with_nano. Expected JSON."
    changes = args.get("changes", {})
    try:
        with open(SCRIPT_PATH, 'r') as f:
            lines = f.readlines()
        for key, value in changes.items():
            if key == 'append':
                lines.append(value + '\n')
            elif key == 'delete_first_line':
                lines = lines[1:]
            else:
                try:
                    line_num = int(key) - 1 # Convert to 0-based index
                    if 0 <= line_num < len(lines):
                        lines[line_num] = value + '\n'
                except ValueError:
                    # If key is not an integer, ignore it
                    pass
        with open(SCRIPT_PATH, 'w') as f:
            f.writelines(lines)
        # Restart the script in a new terminal
        subprocess.Popen(['open', '-a', 'Terminal', sys.executable, SCRIPT_PATH])
        return "Source edited and restarted in new terminal."
    except Exception as e:
        return f"Error editing source: {e}"

# File Reading Tool
def read_and_absorb(args: Union[Dict[str, str], str]) -> str:
    """Read the contents of a file."""
    # Expect args to be a dictionary with 'file_path' key
    if isinstance(args, str):
        # Fallback if LLM provides a string (assume it's the file path)
        args = {"file_path": args}
    file_path = args.get("file_path")
    if not file_path:
        return "Error: Missing 'file_path' argument for read_and_absorb."
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"

# --- Tool Dictionary ---
# This will be updated dynamically by load_cydia_dynamic_tools
tool_dict = {
    # Archive & SSH Tools
    "archive": archive,
    "extract": extract,
    "list_contents": list_contents,
    "convert": convert,
    "manifest": manifest,
    "verify": verify,
    "check_and_fix": check_and_fix,
    "append": append,
    "patch": patch,
    "do_shelly": do_shelly,
    "zharko_tut": zharko_tut,
    "extract_data": extract_data,
    # SEO Tool
    "seo_analyze_site": lambda args: seo_analyze_site(args.get('url') if isinstance(args, dict) else args),
    # GUI Tool
    "recognize_gui_elements": recognize_gui_elements,
    # Shell Tool
    "run_shell_tool": run_shell_tool,
    # Math & Data Processing Tools
    "Calculator": Calculator,
    "train_screen_model": train_screen_model,
    # Image & Screenshot Tools
    "generate_comfy_image": generate_comfy_image,
    "take_screenshot": take_screenshot,
    # Code Analysis & Execution Tools
    "read_and_analyze": read_and_analyze,
    "execute_function": lambda args: execute_function(args.get('filename'), args.get('func_name'), *args.get('args', [])) if isinstance(args, dict) else "Error: Invalid args for execute_function",
    # Source Editing Tool
    "edit_source_with_nano": edit_source_with_nano,
    # File Reading Tool
    "read_and_absorb": read_and_absorb,
}

# --- Cydia Dynamic Tool Loading ---
def load_cydia_dynamic_tools():
    """Load tools defined in Python modules within the Cydia directory."""
    global CYDIA_DYNAMIC_TOOLS, tool_dict
    CYDIA_DYNAMIC_TOOLS.clear()  # Clear previous dynamic tools
    CYDIA_PATH = pathlib.Path(CYDIA_MODULES_PATH)
    if not CYDIA_PATH.exists():
        print(f"[Cydia] Directory {CYDIA_PATH} does not exist. No dynamic tools to load.")
        return
    print("[Cydia] Loading dynamic tools from Python modules...")
    for item in CYDIA_PATH.iterdir():
        if item.is_file() and item.suffix == '.py':
            try:
                # Create a unique module name to avoid conflicts
                module_import_name = f"cydia_dynamic_{item.stem}"
                spec = importlib.util.spec_from_file_location(module_import_name, item)
                if spec is None:
                    print(f"[Cydia] Warning: Could not create spec for {item.name}")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"[*] Loaded Python module: {item.name}")
                # Look for a special attribute or function that defines tools
                # Convention: The module should define a function `register_tools()`
                # that returns a dictionary of {tool_name: callable}
                if hasattr(module, 'register_tools') and callable(module.register_tools):
                    tools_from_module = module.register_tools()
                    if isinstance(tools_from_module, dict):
                        for tool_name, tool_func in tools_from_module.items():
                            if callable(tool_func):
                                CYDIA_DYNAMIC_TOOLS[tool_name] = tool_func
                                print(f"  [Registered Tool] {tool_name}")
                            else:
                                print(f"  [Warning] Tool '{tool_name}' in {item.name} is not callable.")
                    else:
                        print(f"  [Warning] register_tools() in {item.name} did not return a dictionary.")
                else:
                    print(f"  [Info] No 'register_tools' function found in {item.name}. Skipping tool registration.")
            except Exception as e:
                print(f"[Cydia] Error loading module {item.name}: {e}")
                traceback.print_exc()
    # Update the global tool_dict with the newly loaded tools
    # This makes them available to all agents that use tool_dict
    tool_dict.update(CYDIA_DYNAMIC_TOOLS)
    print(f"[Cydia] Finished loading. Total dynamic tools registered: {len(CYDIA_DYNAMIC_TOOLS)}")

#######################################################################################################
# 1. Gemini 1.5 Pro (.gguf via llama.cpp)
#######################################################################################################




# --- Updated load_gemini_model function (Enhanced Agentic Wrapper with Tool Chaining & Streaming) ---
# Assumes global variables BACKEND, MODEL, MODEL_PATH, DYNAMIC_TOOLS are defined in genesis.py
# DYNAMIC_TOOLS should be a dictionary like {'tool_name': function_object}
import os
import json
import inspect
import re # For robust tool call parsing

# --- Utility function for file selection (INCLUDED HERE) ---
def select_gguf_file(initial_dir="./"):
    """Prompts the user to select a .gguf file."""
    gguf_path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        gguf_path = filedialog.askopenfilename(
            title="Select the Gemini 1.5 Pro .gguf file",
            initialdir=initial_dir,
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        root.destroy()
    except (ImportError, Exception): # Catch general exceptions for robustness
        print("GUI file dialog not available. Please enter the path manually.")
    if not gguf_path:
        print(f"\nEnter the full path to your Gemini 1.5 Pro .gguf file.")
        print(f"(e.g., /Users/username/Models/gemini-1.5-pro-Q4_K_M.gguf)")
        gguf_path = input("Path: ").strip()
    if not gguf_path or not os.path.isfile(gguf_path):
        print(f"[ERROR] Invalid file path: {gguf_path}")
        raise FileNotFoundError(f"Could not find GGUF file at {gguf_path}")
    if not gguf_path.lower().endswith('.gguf'):
        print(f"[WARNING] Selected file does not have a .gguf extension: {gguf_path}")
    return gguf_path

def load_gemini_model():
    global BACKEND, MODEL, MODEL_PATH, DYNAMIC_TOOLS# Ensure DYNAMIC_TOOLS is global
    BACKEND = "llama"
    GEMINI_MODEL_NAME = "Gemini 1.5 Pro (.gguf via llama.cpp)"

    # --- 1. Ensure MODEL_PATH is set (Prompt User if Not) ---
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        print(f"\n--- Loading {GEMINI_MODEL_NAME} ---")
        try:
            MODEL_PATH = select_gguf_file(initial_dir="./")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
    print(f"[INFO] Using model file: {MODEL_PATH}")

    # --- 2. Prompt User for Hardware Utilization Mode ---
    print("\nSelect hardware utilization mode for loading the model:")
    print("1. CPU Only (Maximum compatibility, disables Metal backend)")
    print("2. Metal GPU Accelerated (Fast performance, uses VRAM)")
    print("3. Metal GPU with CPU Fallback (Uses GPU if possible, falls back to CPU if it fails)")
    mode_choice = input("Enter choice (1/2/3): ").strip()

    # --- 3. Set Environment Variables BEFORE ANY llama_cpp import ---
    # Critical for backend control and potential performance tweaks
    if mode_choice == '1':
        print("\n[Mode Selected] CPU Only - Setting GGML_METAL_DISABLE=1.")
        os.environ['GGML_METAL_DISABLE'] = '1'
        # Potentially disable BLAS for simpler CPU path if issues arise (uncomment if needed)
        # os.environ.pop('GGML_USE_BLAS', None)
    else:
        os.environ.pop('GGML_METAL_DISABLE', None)
        # Consider enabling BLAS for CPU portions in GPU fallback mode (uncomment if BLAS is installed/configured)
        # os.environ['GGML_USE_BLAS'] = '1'

    # --- 4. Import llama_cpp (after env var setting) ---
    try:
        from llama_cpp import Llama
        print("[INFO] llama-cpp-python is available.")
    except ImportError:
        print("[ERROR] llama-cpp-python not found.")
        print("Please install it manually with Metal support:")
        print("CMAKE_ARGS='-DGGML_METAL=on -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64' pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
        return

    # --- 5. Define Optimized Configuration Parameters ---
    # --- User Input for Context Size ---
    MAX_CTX = 32768 # Conservative safety cap for stability, especially on macOS
    DEFAULT_CTX = 8192
    print(f"\n--- Context Size Configuration (Max {MAX_CTX}) ---")
    print(f"The model supports up to 131072 tokens. Adjust based on needs/resources.")
    try:
        user_ctx_input = input(f"Enter desired context size (n_ctx) (default {DEFAULT_CTX}, max {MAX_CTX}): ").strip()
        if user_ctx_input:
            CONTEXT_SIZE = int(user_ctx_input)
        else:
            CONTEXT_SIZE = DEFAULT_CTX
    except ValueError:
        print(f"[WARNING] Invalid input. Using default context size {DEFAULT_CTX}.")
        CONTEXT_SIZE = DEFAULT_CTX
    if CONTEXT_SIZE > MAX_CTX:
        print(f"[WARNING] Requested context size {CONTEXT_SIZE} exceeds recommended maximum {MAX_CTX}. Capping at {MAX_CTX}.")
        CONTEXT_SIZE = MAX_CTX
    if CONTEXT_SIZE <= 0:
        print(f"[WARNING] Invalid context size. Using default {DEFAULT_CTX}.")
        CONTEXT_SIZE = DEFAULT_CTX

    # --- Optimized Parameters Based on Mode and Common Practices ---
    # Batch Size: Larger batches improve throughput, capped at context size
    BATCH_SIZE = min(512, CONTEXT_SIZE) # Using 512 as a common efficient batch size per user reports
    # Threads: Use all available threads for maximum throughput (common optimization)
    NUM_THREADS = max(1, os.cpu_count() or 1)
    CPU_FALLBACK_THREADS = NUM_THREADS # Use all for fallback too
    GPU_LAYERS_MAX = 99 # 99 typically means "use as many as possible"

    # --- 6. Loading Logic Based on User Choice ---
    llm_instance = None
    load_success = False
    try:
        if mode_choice == '1':
            # --- CPU Only Mode (Optimized) ---
            print(f"\n[LOADING] Attempting to load model with parameters for CPU ONLY (n_ctx={CONTEXT_SIZE}):")
            CPU_CTX_SIZE = CONTEXT_SIZE
            params = {
                "model_path": MODEL_PATH,
                "n_ctx": CPU_CTX_SIZE,
                "n_batch": BATCH_SIZE,
                "n_threads": CPU_FALLBACK_THREADS,
                "n_threads_batch": CPU_FALLBACK_THREADS, # Dedicated batch threads (llama.cpp >= 0.2.72)
                "n_gpu_layers": 0,
                "use_mlock": False, # Can cause issues on some systems, disable by default
                "flash_attn": False, # Often not beneficial or unstable on CPU
                "logits_all": False,
                "embedding": False,
                "verbose": False, # Reduce logs
                # Consider adding numa=False if NUMA issues are reported (uncomment if needed)
                # "numa": False
            }
            print("Parameters:")
            for k, v in params.items():
                if k != "model_path": print(f"  - {k}: {v}")
            llm_instance = Llama(**params)
            load_success = True
            print(f"[SUCCESS] Model loaded successfully on CPU with n_ctx={CPU_CTX_SIZE}.")
            print("[INFO] Metal backend should be DISABLED for computation.")

        elif mode_choice in ['2', '3']: # Combine GPU modes
             # --- Metal GPU Modes (with potential CPU Fallback for mode 3) ---
             FALLBACK_TO_CPU_ON_FAILURE = (mode_choice == '3')
             print(f"\n[LOADING] Attempting to load model with parameters for Metal GPU (Primary Attempt, n_ctx={CONTEXT_SIZE}):")
             params_gpu = {
                 "model_path": MODEL_PATH,
                 "n_ctx": CONTEXT_SIZE,
                 "n_batch": BATCH_SIZE,
                 "n_threads": NUM_THREADS, # Threads for data prep/offload
                 "n_threads_batch": NUM_THREADS, # Dedicated batch threads
                 "n_gpu_layers": GPU_LAYERS_MAX,
                 "use_mlock": False, # Disable mlock for stability
                 "flash_attn": True, # Enable for potential speed gains on supported hardware
                 "logits_all": False,
                 "embedding": False,
                 "verbose": False,
                 "type_k": 8, "type_v": 8 # KV cache Q8_0: Commonly used for VRAM savings/speed on Metal
                 # Consider tensor_split if multi-GPU issues or wanting to control distribution (uncomment if needed)
                 # "tensor_split": [1.0] # Example for single GPU
             }
             print("Parameters (GPU Attempt):")
             for k, v in params_gpu.items():
                 if k not in ["model_path"]: print(f"  - {k}: {v}")
             try:
                 llm_instance = Llama(**params_gpu)
                 load_success = True
                 actual_gpu_layers = getattr(llm_instance, 'n_gpu_layers', 'Unknown')
                 print(f"[SUCCESS] Model loaded successfully on Metal (Primary Attempt).")
                 print(f"  - Requested Context Size: {CONTEXT_SIZE} tokens")
                 print(f"  - Batch Size: {BATCH_SIZE}")
                 print(f"  - Requested GPU Layers: {GPU_LAYERS_MAX} (Max)")
                 if actual_gpu_layers != 'Unknown':
                     print(f"  - Actual GPU Layers Offloaded: {actual_gpu_layers}")
                 print("[INFO] Ensure your prompt uses the correct chat template.")
             except Exception as e_gpu:
                 print(f"[ERROR] Failed to load Llama model with Metal GPU: {e_gpu}")
                 if FALLBACK_TO_CPU_ON_FAILURE:
                     print("\n[LOADING] Falling back to CPU loading...")
                     print("Setting GGML_METAL_DISABLE=1 for fallback...")
                     os.environ['GGML_METAL_DISABLE'] = '1'
                     CPU_CTX_SIZE = min(CONTEXT_SIZE, 8192) # Cap context for fallback
                     params_cpu = {
                         "model_path": MODEL_PATH,
                         "n_ctx": CPU_CTX_SIZE,
                         "n_batch": BATCH_SIZE,
                         "n_threads": CPU_FALLBACK_THREADS,
                         "n_threads_batch": CPU_FALLBACK_THREADS, # Dedicated batch threads
                         "n_gpu_layers": 0,
                         "use_mlock": False,
                         "flash_attn": False,
                         "logits_all": False,
                         "embedding": False,
                         "verbose": False
                     }
                     print("Parameters (CPU Fallback):")
                     for k, v in params_cpu.items():
                         if k != "model_path": print(f"  - {k}: {v}")
                     llm_instance = Llama(**params_cpu)
                     load_success = True
                     print(f"[SUCCESS] Model loaded successfully on CPU (Fallback) with n_ctx={CPU_CTX_SIZE}.")
                     print("[INFO] Metal backend should be DISABLED for fallback computation.")
                 else:
                     return
        else:
            print("[ERROR] Invalid hardware mode choice. Please select 1, 2, or 3.")
            return

    except Exception as e:
        print(f"[CRITICAL ERROR] Model loading failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 7. Final Assignment and Confirmation ---
    if load_success and llm_instance is not None:
        MODEL = llm_instance
        print(f"\n[INFO] Global MODEL variable assigned successfully.")
        print(f"[INFO] Model '{GEMINI_MODEL_NAME}' is ready for use.")
    else:
        print("[CRITICAL ERROR] Model loading process did not complete successfully.")
        return

    # --- 8. Enhanced Agentic Interaction Loop with Planning & Tool Chaining ---
    def run_gemini_agent_loop():
        if MODEL is None or not callable(MODEL):
             print(f"[ERROR] Loaded MODEL object is not callable or is None. Type: {type(MODEL) if MODEL else 'None'}")
             return

        # --- Tool Description ---
        def get_tools_description():
            if not DYNAMIC_TOOLS:
                return "No external tools are available."
            descriptions = []
            for name, func in DYNAMIC_TOOLS.items():
                sig = inspect.signature(func)
                params_desc = []
                for param_name, param in sig.parameters.items():
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'any'
                    default_val = param.default if param.default != inspect.Parameter.empty else None
                    param_desc = f"  - {param_name} ({param_type})"
                    if default_val is not None:
                        param_desc += f" [default: {default_val}]"
                    params_desc.append(param_desc)
                func_desc = f"Function: {name}\nDescription: {func.__doc__ or 'No description provided.'}\nParameters:\n" + "\n".join(params_desc)
                descriptions.append(func_desc)
            return "\n\n".join(descriptions)

        tools_desc = get_tools_description()

        # --- Enhanced System Prompt for Agentic Behavior ---
        system_prompt = (
            "You are a highly capable AI agent named Gemini 1.5 Pro Agent. "
            "Your goal is to be helpful, thorough, and autonomous in completing user requests.\n\n"
            "You have access to the following tools:\n"
            f"{tools_desc}\n\n"
            "To use one or more tools, respond EXACTLY with a JSON array containing the tool calls:\n"
            "[{\"name\": \"<function_name>\", \"arguments\": {\"<arg1>\": \"<value1>\"}}, {\"name\": \"<function2>\"}]\n"
            "You can request multiple tools to be executed in sequence.\n"
            "After executing tools, you will receive their results. Use these results to formulate your final answer.\n"
            "If no tools are needed, respond directly to the user's query.\n"
            "Think step-by-step if necessary. Break down complex tasks. "
            "If you cannot complete a task, explain why clearly."
        )

        # Initialize conversation history with system prompt
        conversation_history = [{"role": "system", "content": system_prompt}]
        print("\n--- Gemini 1.5 Pro Agent (Streaming - Type 'exit' to quit) ---")
        print("[INFO] Agent can plan, chain tools, and stream responses.")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Agent: Goodbye!")
                    break

                conversation_history.append({"role": "user", "content": user_input})

                # --- Efficient Prompt Formatting ---
                formatted_prompt = "<|begin_of_text|>"
                for message in conversation_history:
                    role = message["role"]
                    content = message["content"]
                    formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
                formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

                # --- Fast Generation with Streaming ---
                # Optimized generation parameters for responsiveness and quality
                completion_kwargs = {
                    "prompt": formatted_prompt,
                    "max_tokens": 1500, # Balanced default
                    "temperature": 0.7, # Slightly creative but focused
                    "top_p": 0.9, # Nucleus sampling
                    "min_p": 0.05, # Helps prevent low probability tokens (user suggestion)
                    "repeat_penalty": 1.1, # Slight penalty for repetition
                    "echo": False,
                    "stream": True, # KEY: Enable streaming for responsiveness
                    "stop": ["<|eot_id|>"],
                }

                print("Agent: ", end='', flush=True)
                response_text = ""
                full_response = MODEL(**completion_kwargs)

                # --- Stream and Collect Response ---
                for chunk in full_response:
                    delta = ""
                    if isinstance(chunk, dict) and 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        if 'text' in choice:
                            delta = choice.get('text', '')
                        elif 'delta' in choice and 'content' in choice['delta']:
                             delta = choice['delta'].get('content', '')
                    if delta:
                        print(delta, end='', flush=True)
                        response_text += delta
                print() # Newline after stream

                # --- Process Potential Tool Calls (Robust Parsing) ---
                tool_results = []
                try:
                    match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        tool_calls_data = json.loads(json_str)
                        if isinstance(tool_calls_data, list):
                            for i, tool_call_data in enumerate(tool_calls_data):
                                if isinstance(tool_call_data, dict) and 'name' in tool_call_data and 'arguments' in tool_call_data:
                                    tool_name = tool_call_data['name']
                                    tool_args = tool_call_data.get('arguments', {})
                                    if tool_name in DYNAMIC_TOOLS:
                                        print(f"\n[INFO] Calling tool ({i+1}/{len(tool_calls_data)}): {tool_name}")
                                        try:
                                            tool_func = DYNAMIC_TOOLS[tool_name]
                                            tool_result = tool_func(**tool_args)
                                            print(f"[INFO] Tool '{tool_name}' result: {tool_result}")
                                            tool_results.append({"name": tool_name, "result": tool_result})
                                        except Exception as tool_e:
                                            error_msg = f"Error executing '{tool_name}': {tool_e}"
                                            print(f"[ERROR] {error_msg}")
                                            tool_results.append({"name": tool_name, "result": error_msg})
                                    else:
                                        print(f"[WARNING] Agent requested unknown tool: {tool_name}")
                                        tool_results.append({"name": tool_name, "result": f"Unknown tool '{tool_name}'. Available tools: {list(DYNAMIC_TOOLS.keys())}"})
                        else:
                            pass # Not a list of calls
                    # else: # No JSON array found, treat as direct response
                except json.JSONDecodeError:
                    pass # Not JSON, treat as direct response
                except Exception as e:
                    print(f"[ERROR] Error during tool call processing: {e}")

                # --- Handle Tool Results and Generate Final Answer ---
                final_response_text = response_text
                if tool_results:
                    conversation_history.append({"role": "assistant", "content": response_text})
                    results_str = "\n".join([f"Result for '{tr['name']}': {tr['result']}" for tr in tool_results])
                    conversation_history.append({"role": "tool", "content": results_str})

                    # --- Re-prompt Model with Tool Results (Streamed) ---
                    tool_prompt = "<|begin_of_text|>"
                    for message in conversation_history:
                        r = message["role"]
                        c = message["content"]
                        tool_prompt += f"<|start_header_id|>{r}<|end_header_id|>\n{c}<|eot_id|>"
                    tool_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

                    tool_kwargs = {**completion_kwargs, "prompt": tool_prompt, "max_tokens": 1500}
                    print("Agent (Final Answer): ", end='', flush=True)
                    final_response_text = ""
                    for chunk in MODEL(**tool_kwargs):
                        delta = ""
                        if isinstance(chunk, dict) and 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            if 'text' in choice:
                                delta = choice.get('text', '')
                            elif 'delta' in choice and 'content' in choice['delta']:
                                 delta = choice['delta'].get('content', '')
                        if delta:
                            print(delta, end='', flush=True)
                            final_response_text += delta
                    print()
                    conversation_history.append({"role": "assistant", "content": final_response_text.strip()})
                else:
                    # If no tools, just add the initial streamed response to history
                    conversation_history.append({"role": "assistant", "content": final_response_text.strip()})

            except KeyboardInterrupt:
                print("\nAgent: Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] An error occurred during interaction: {e}")
                import traceback
                traceback.print_exc()

    # Call the enhanced agent loop function after successful loading
    run_gemini_agent_loop()

# --- End of load_gemini_model function ---




































#######################################################################################################
# 2. Deepseek R1 7B (via ollama)
#######################################################################################################
def load_deepseek_model():
    global BACKEND, MODEL
    BACKEND = "ollama"
    MODEL = DEEPSEEK_MODEL
    if not command_exists('ollama'):
        print("Ollama not found. Please install Ollama first: https://ollama.com/download")
        sys.exit(1)
    # Pull the model if not present (ollama will handle this)
    try:
        subprocess.check_call(['ollama', 'show', MODEL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Pulling Ollama model: {MODEL}")
        subprocess.check_call(['ollama', 'pull', MODEL])

#######################################################################################################
# 3. Qwen (.safetensors via transformers)
#######################################################################################################




def load_qwen_model():
    global BACKEND, MODEL, MODEL_DIR
    BACKEND = "transformers"
    MODEL_DIR = QWEN_PATH  # This should point to the downloaded model directory or HF repo
    if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
        print("Qwen model path (QWEN_PATH) not set or invalid.")
        choice = input("Enter 1 to provide local path to model directory, 2 to load from Hugging Face (Qwen/Qwen3-8B): ").strip()
        if choice == '1':
            local_path = input("Enter the path to the model directory (containing config.json, tokenizer files, and all model-*.safetensors files): ").strip()
            if os.path.isfile(local_path) and local_path.endswith('.safetensors'):
                local_path = os.path.dirname(os.path.abspath(local_path))
            if not os.path.isdir(local_path):
                print("Invalid directory path provided.")
                sys.exit(1)
            MODEL_DIR = local_path
        elif choice == '2':
            MODEL_DIR = "Qwen/Qwen3-8B"
        else:
            print("Invalid choice.")
            sys.exit(1)

    # Check for config.json if local directory
    if os.path.isdir(MODEL_DIR):
        if not os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
            print("Error: The model directory must contain 'config.json'. Please ensure you have downloaded the full model files including all safetensors shards and configuration.")
            download_choice = input("Would you like to download the Qwen3-8B model now? (y/n): ").strip().lower()
            if download_choice == 'y':
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    print("Installing huggingface_hub...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                    from huggingface_hub import snapshot_download
                print(f"Downloading Qwen3-8B to {MODEL_DIR}... (this may take a while, ~16GB)")
                snapshot_download(repo_id="Qwen/Qwen3-8B", local_dir=MODEL_DIR, ignore_patterns=["*.gitattributes", "README.md"])
                print(f"Model successfully downloaded to {MODEL_DIR}")
            else:
                print("Cannot proceed without valid model files.")
                sys.exit(1)

        # Check for model weight files
        def has_model_weights(dir_path):
            # Sharded checkpoints (common for large models like Qwen3-8B)
            import glob
            if glob.glob(os.path.join(dir_path, 'model-*-of-*.safetensors')):
                return True
            # Index file
            if os.path.exists(os.path.join(dir_path, 'model.safetensors.index.json')):
                return True
            # Fallback single file (unlikely for 8B model)
            single_files = ['model.safetensors', 'pytorch_model.bin']
            for f in single_files:
                if os.path.exists(os.path.join(dir_path, f)):
                    return True
            return False

        if not has_model_weights(MODEL_DIR):
            print("Error: No model weight files found (e.g., 'model-00001-of-000XX.safetensors' and 'model.safetensors.index.json'). The model cannot load without these.")
            download_choice = input("Would you like to download the Qwen3-8B model now? (y/n): ").strip().lower()
            if download_choice == 'y':
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    print("Installing huggingface_hub...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                    from huggingface_hub import snapshot_download
                print(f"Downloading Qwen3-8B to {MODEL_DIR}... (this may take a while, ~16GB)")
                snapshot_download(repo_id="Qwen/Qwen3-8B", local_dir=MODEL_DIR, ignore_patterns=["*.gitattributes", "README.md"])
                print(f"Model successfully downloaded to {MODEL_DIR}")
            else:
                print("Cannot proceed without valid model files.")
                sys.exit(1)

    # Ensure libraries are installed (as before)
    try:
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if tuple(map(int, transformers.__version__.split('.')[:2])) < (4, 37):
            print(f"Warning: transformers version {transformers.__version__} is outdated. Upgrading to >=4.37.0.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.37.0"])
    except ImportError:
        print("Installing transformers library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.37.0"])
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    try:
        import accelerate
    except ImportError:
        print("Installing accelerate library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
        import accelerate

    try:
        import torch
    except ImportError:
        print("Installing torch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        import torch

    if torch.cuda.is_available():
        try:
            import bitsandbytes
        except ImportError:
            print("Installing bitsandbytes for quantization...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
            import bitsandbytes
    else:
        print("CUDA not available. Loading model without 4-bit quantization.")

    try:
        print(f"Loading Qwen tokenizer from '{MODEL_DIR}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Fix pad/EOS warning

        print(f"Loading Qwen model from '{MODEL_DIR}'...")
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Quantized Qwen model loaded successfully with 4-bit quantization.")
        else:
            dtype = torch.float16
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            print(f"Qwen model loaded successfully on {device} with dtype {dtype}.")

        model.config.pad_token_id = tokenizer.pad_token_id

        MODEL = {
            "model": model,
            "tokenizer": tokenizer,
            "type": "transformers"
        }

    except Exception as e:
        print(f"Error loading model from '{MODEL_DIR}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)






































#######################################################################################################
# 4. N64 Dev Agent
#######################################################################################################






def launch_n64_agent():
    """Wrapper function to launch the N64 Dev Agent."""
    # Check if a .z64 file was dragged onto the script
    rom_path_arg = None
    if len(sys.argv) > 1:
        potential_rom = sys.argv[1]
        if os.path.isfile(potential_rom) and potential_rom.lower().endswith('.z64'):
            rom_path_arg = potential_rom
            print(f"[*] Found .z64 file argument: {rom_path_arg}")
    # Launch the N64 Dev Agent
    n64dev_agent(rom_path_arg) # This function now returns control
    # Signal that the agent has finished its task
    # It's crucial NOT to call sys.exit(0) here if you want to return to the main menu.
    # The agent should return control to the main script's menu loop.
    print("[*] Returned from N64 Dev Agent.")


import os
import sys
import subprocess
import shutil
import stat
import traceback
import zipfile
# import tarfile # <-- REMOVED: No longer needed for ISL tar.gz
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List
# --- Helper function for checking command existence ---
def command_exists(cmd):
    """Check if a command exists using shutil.which."""
    return shutil.which(cmd) is not None
# --- N64 Dev Agent (Full Code - Updated for 2-Stage GCC, ISL REMOVED, Robust Handling) ---
def n64dev_agent(rom_path_override: Optional[str] = None):
    """N64 Development Agent using custom n64dev/Libdragon from user's repo."""
    # Store original directory to return to later
    ORIGINAL_CWD_AGENT = Path.cwd()
    WORK = Path.home() / ".n64dev"
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK) # Change to the working directory for the agent's session
    # --- N64 Dev Environment Setup Variables ---
    zip_urls = [
        "https://github.com/Rostyslav296/n64dev/raw/main/N64Recomp-mod-tool-release.zip",
        # "https://github.com/Rostyslav296/n64dev/raw/main/gcc-toolchain-mips64-latest.zip", # EXCLUDED
        "https://github.com/Rostyslav296/n64dev/raw/main/gmp-6.3.0.zip",
        "https://github.com/Rostyslav296/n64dev/raw/main/libdragon-toolchain-continuous-prerelease.zip",
        "https://github.com/Rostyslav296/n64dev/raw/main/mpc-1.3.1.zip",
        "https://github.com/Rostyslav296/n64dev/raw/main/mpfr-4.2.2.zip",
        "https://github.com/Rostyslav296/n64dev/raw/main/n64chain-9.1.0%202.zip",
        "https://github.com/Rostyslav296/n64dev/raw/main/newlib-cygwin-3.6.4.zip"
    ]
    # --- REMOVED: ISL tar.gz URL ---
    # tar_gz_urls = [
    #     "https://ftp.gnu.org/gnu/isl/isl-0.26.tar.gz"
    # ]
    # --- Centralized SDK Directory ---
    N64SDK_DIR = os.path.join(WORK, "n64sdk")
    os.makedirs(N64SDK_DIR, exist_ok=True)
    N64_INST = os.path.join(N64SDK_DIR, "install")
    BINUTILS_DIR = os.path.join(N64SDK_DIR, "binutils-gdb")
    GCC_SOURCE_DIR = os.path.join(N64SDK_DIR, "gcc-source")
    # ISL_DIR = os.path.join(N64SDK_DIR, "isl-0.26") # <-- REMOVED: ISL directory
    os.environ["N64_INST"] = N64_INST
    # --- Improved find_source_root to search recursively ---
    def find_source_root(extract_dir: str) -> str:
        """Finds the source root directory recursively."""
        for root, dirs, files in os.walk(extract_dir):
            if 'configure' in files:
                print(f"[*] Found 'configure' in subdirectory: {root}")
                return root
        print(f"[!] Warning: Could not find definitive source root in '{extract_dir}'. Assuming '{extract_dir}' is correct.")
        return extract_dir
    def is_build_complete(component_name: str, build_dir: str, install_dir: str) -> bool:
        """Checks if a component's build/install appears complete."""
        if not os.path.exists(build_dir):
            return False
        if component_name == "gmp-6.3.0":
            return os.path.isfile(os.path.join(install_dir, "lib", "libgmp.a"))
        elif component_name == "mpfr-4.2.2":
            return os.path.isfile(os.path.join(install_dir, "lib", "libmpfr.a"))
        elif component_name == "mpc-1.3.1":
            return os.path.isfile(os.path.join(install_dir, "lib", "libmpc.a"))
        elif component_name == "binutils-gdb":
            return os.path.isfile(os.path.join(install_dir, "bin", "mips64-elf-as"))
        # elif component_name == "isl-0.26": # <-- REMOVED: ISL check
        #     return os.path.isfile(os.path.join(install_dir, "lib", "libisl.a"))
        elif component_name == "gcc-stage1":
            gcc_bin = os.path.join(install_dir, "bin", "mips64-elf-gcc")
            return os.path.isfile(gcc_bin) # Basic check
        elif component_name == "gcc-stage2":
            gcc_bin = os.path.join(install_dir, "bin", "mips64-elf-gcc")
            gxx_bin = os.path.join(install_dir, "bin", "mips64-elf-g++")
            return os.path.isfile(gcc_bin) and os.path.isfile(gxx_bin)
        elif component_name == "newlib-cygwin-3.6.4":
            return os.path.isfile(os.path.join(install_dir, "mips64-elf", "include", "sys", "types.h"))
        elif component_name == "libdragon-toolchain-continuous-prerelease":
            return os.path.isfile(os.path.join(install_dir, "mips64-elf", "lib", "libdragon.a"))
        print(f"[!] No specific completion check defined for '{component_name}'. Assuming incomplete.")
        return False
    def verify_install(component_name: str, build_dir: str, install_dir: str) -> bool:
        """Verifies if the component is properly installed after build."""
        if is_build_complete(component_name, build_dir, install_dir):
            print(f"[+] Post-install verification successful for {component_name}.")
            return True
        else:
            print(f"[!] Post-install verification failed for {component_name}. Check installation contents.")
            return False
    # def download_and_extract_tar_gz(url: str, extract_to: str): # <-- REMOVED: Helper for tar.gz (ISL)
    #     """Downloads and extracts a .tar.gz file."""
    #     filename = url.split('/')[-1]
    #     local_tar_path = os.path.join(extract_to, filename)
    #     extract_dir_name = filename.replace('.tar.gz', '')
    #     final_extract_dir = os.path.join(extract_to, extract_dir_name)
    #     print(f"[*] Downloading {filename} from {url}...")
    #     try:
    #         urllib.request.urlretrieve(url, local_tar_path)
    #         print(f"[+] Downloaded {filename} successfully.")
    #     except Exception as e:
    #         print(f"[!] Failed to download {filename}: {e}")
    #         traceback.print_exc()
    #         if os.path.exists(local_tar_path): os.remove(local_tar_path)
    #         return False
    #     print(f"[*] Extracting {filename}...")
    #     try:
    #         with tarfile.open(local_tar_path, 'r:gz') as tar_ref:
    #             tar_ref.extractall(extract_to)
    #         print(f"[+] Extracted {filename} to {final_extract_dir}")
    #         if os.path.exists(final_extract_dir): return final_extract_dir
    #         else:
    #             print(f"[!] Extraction seemed successful, but directory {final_extract_dir} not found.")
    #             return False
    #     except Exception as e:
    #         print(f"[!] Failed to extract {filename}: {e}")
    #         traceback.print_exc()
    #         return False
    #     finally:
    #         if os.path.exists(local_tar_path): os.remove(local_tar_path)
    def prompt_for_source(component_name_display: str, expected_file: str = "configure"):
        """Prompts user for source directory."""
        print(f"[*] {component_name_display} is required. Please provide the source folder.")
        if "GCC" in component_name_display:
             print("    Download GCC 9.1.0 source from: https://ftp.gnu.org/gnu/gcc/gcc-9.1.0/gcc-9.1.0.tar.gz")
        src_path = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            src_path = filedialog.askdirectory(title=f"Select the {component_name_display} source folder")
            root.destroy()
        except Exception as e:
            print(f"[!] GUI dialog failed or cancelled: {e}")
        if not src_path or not os.path.isdir(src_path):
            src_path = input(f"Enter the full path to the {component_name_display} source folder: ").strip()
        expected_path = os.path.join(src_path, expected_file)
        if os.path.isdir(src_path) and os.path.isfile(expected_path):
            print(f"[*] Valid source selected for {component_name_display}: {src_path}")
            return src_path
        else:
            print(f"[!] Invalid source for {component_name_display}: {src_path} or '{expected_file}' missing.")
            return None
    def setup_chain():
        """Sets up the N64 toolchain and Libdragon."""
        print("[*] Setting up N64 toolchain and Libdragon...")
        print(f"    N64SDK Directory: {N64SDK_DIR}")
        print(f"    N64_INST (Install Prefix): {N64_INST}")
        os.makedirs(N64_INST, exist_ok=True)
        # --- 1. REMOVED: Handle ISL Download/Extract/Build ---
        # if not is_build_complete("isl-0.26", ISL_DIR, N64_INST):
        #     print("[*] ISL 0.26 build required.")
        #     if os.path.exists(ISL_DIR): shutil.rmtree(ISL_DIR)
        #     extracted_dir = download_and_extract_tar_gz(tar_gz_urls[0], N64SDK_DIR)
        #     if not extracted_dir or not os.path.exists(extracted_dir):
        #         print("[!] Failed to download/extract ISL. Prompting for source.")
        #         isl_src = prompt_for_source("ISL 0.26")
        #         if not isl_src: return False
        #         if os.path.exists(ISL_DIR): shutil.rmtree(ISL_DIR)
        #         shutil.move(isl_src, ISL_DIR)
        #     else:
        #         if os.path.abspath(extracted_dir) != os.path.abspath(ISL_DIR):
        #             if os.path.exists(ISL_DIR): shutil.rmtree(ISL_DIR)
        #             shutil.move(extracted_dir, ISL_DIR)
        #             print(f"[+] Moved ISL source to {ISL_DIR}.")
        #     os.chdir(ISL_DIR)
        #     print(f"[*] Building ISL in {ISL_DIR}...")
        #     try:
        #         configure_script_path = "./configure"
        #         if os.path.exists(configure_script_path):
        #             st = os.stat(configure_script_path)
        #             os.chmod(configure_script_path, st.st_mode | stat.S_IEXEC)
        #         configure_cmd = ["./configure", f"--prefix={N64_INST}", f"--with-gmp-prefix={N64_INST}"]
        #         print(f"[CMD] {' '.join(configure_cmd)}")
        #         subprocess.check_call(configure_cmd)
        #         make_cmd = ["make", "-j4"]
        #         print(f"[CMD] {' '.join(make_cmd)}")
        #         subprocess.check_call(make_cmd)
        #         make_install_cmd = ["make", "install"]
        #         print(f"[CMD] {' '.join(make_install_cmd)}")
        #         subprocess.check_call(make_install_cmd)
        #         print("[+] ISL built and installed successfully.")
        #         if not verify_install("isl-0.26", ISL_DIR, N64_INST):
        #             return False
        #     except subprocess.CalledProcessError as e:
        #         print(f"[!] Error building ISL: {e}")
        #         with open(os.path.join(WORK, "isl_build_log.txt"), "w") as log_file:
        #             log_file.write(str(e) + "\n" + (e.output.decode() if e.output else ""))
        #         print("[!] Detailed logs saved to isl_build_log.txt")
        #         return False
        #     except Exception as e:
        #         print(f"[!] Unexpected error building ISL: {e}")
        #         traceback.print_exc()
        #         with open(os.path.join(WORK, "isl_build_log.txt"), "w") as log_file:
        #             log_file.write(traceback.format_exc())
        #         print("[!] Detailed logs saved to isl_build_log.txt")
        #         return False
        #     finally:
        #         os.chdir(WORK)
        # else:
        #     print("[*] ISL 0.26 build appears complete. Skipping.")
        print("[*] ISL support has been removed from this build script.")
        # --- 2. Handle Other ZIP Components (excluding GCC zip) ---
        filtered_zip_urls = [url for url in zip_urls if "gcc-toolchain-mips64-latest" not in url]
        for url in filtered_zip_urls:
            filename = url.split('/')[-1].replace('%20', ' ')
            intended_extract_name = filename.replace('.zip', '')
            final_extract_dir = os.path.join(N64SDK_DIR, intended_extract_name)
            if is_build_complete(intended_extract_name, final_extract_dir, N64_INST):
                print(f"[*] Build for '{intended_extract_name}' appears complete. Skipping.")
                continue
            print(f"[*] Setting up '{intended_extract_name}'...")
            if os.path.exists(final_extract_dir): shutil.rmtree(final_extract_dir)
            download_success = False
            print(f"[*] Downloading {filename}...")
            local_zip_path = os.path.join(N64SDK_DIR, filename)
            try:
                urllib.request.urlretrieve(url, local_zip_path)
                print(f"[+] Downloaded {filename}.")
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    temp_extract = os.path.join(N64SDK_DIR, f"temp_extract_{intended_extract_name}")
                    if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
                    os.makedirs(temp_extract, exist_ok=True)
                    zip_ref.extractall(temp_extract)
                print(f"[+] Extracted {filename}.")
                actual_root = find_source_root(temp_extract)
                shutil.move(actual_root, final_extract_dir)
                print(f"[+] Moved source to {final_extract_dir}.")
                download_success = True
            except Exception as e:
                print(f"[!] Error with {filename}: {e}")
                traceback.print_exc()
                with open(os.path.join(WORK, f"{intended_extract_name}_download_log.txt"), "w") as log_file:
                    log_file.write(traceback.format_exc())
                print(f"[!] Detailed logs saved to {intended_extract_name}_download_log.txt")
            finally:
                if os.path.exists(local_zip_path): os.remove(local_zip_path)
                temp_cleanup = os.path.join(N64SDK_DIR, f"temp_extract_{intended_extract_name}")
                if os.path.exists(temp_cleanup): shutil.rmtree(temp_cleanup)
            if not download_success:
                print(f"[!] Failed to get '{intended_extract_name}'. Prompting for source.")
                user_src = prompt_for_source(f"'{intended_extract_name}'")
                if not user_src: return False
                if os.path.exists(final_extract_dir): shutil.rmtree(final_extract_dir)
                shutil.move(user_src, final_extract_dir)
                print(f"[+] Source for '{intended_extract_name}' acquired.")
        # --- 3. Handle Binutils (Manual Upload) ---
        if not is_build_complete("binutils-gdb", BINUTILS_DIR, N64_INST):
            print("[*] Binutils build required.")
            if os.path.exists(BINUTILS_DIR): shutil.rmtree(BINUTILS_DIR)
            binutils_src = prompt_for_source("Binutils")
            if not binutils_src: return False
            if os.path.exists(BINUTILS_DIR): shutil.rmtree(BINUTILS_DIR)
            shutil.move(binutils_src, BINUTILS_DIR)
            if not os.path.isfile(os.path.join(BINUTILS_DIR, "configure")):
                print(f"[!] Error: 'configure' not found in {BINUTILS_DIR}.")
                return False
            print(f"[+] Binutils source acquired at {BINUTILS_DIR}.")
        else:
            print("[*] Binutils build appears complete. Skipping.")
        # --- 4. Handle GCC Source (Manual Upload) ---
        if not os.path.isfile(os.path.join(GCC_SOURCE_DIR, "configure")):
            print("[*] GCC source is required.")
            print("    Download GCC 9.1.0 source from: https://ftp.gnu.org/gnu/gcc/gcc-9.1.0/gcc-9.1.0.tar.gz")
            if os.path.exists(GCC_SOURCE_DIR): shutil.rmtree(GCC_SOURCE_DIR)
            gcc_src = prompt_for_source("GCC 9.1.0")
            if not gcc_src: return False
            if os.path.exists(GCC_SOURCE_DIR): shutil.rmtree(GCC_SOURCE_DIR)
            shutil.move(gcc_src, GCC_SOURCE_DIR)
            if not os.path.isfile(os.path.join(GCC_SOURCE_DIR, "configure")):
                print(f"[!] Error: 'configure' not found in {GCC_SOURCE_DIR}.")
                return False
            # Patch for MacOS fdopen conflict in zlib
            zutil_path = os.path.join(GCC_SOURCE_DIR, 'zlib', 'zutil.h')
            if os.path.exists(zutil_path):
                with open(zutil_path, 'r') as f:
                    content = f.read()
                content = content.replace('#    define fdopen(fd,mode) NULL /* No fdopen() */', '')
                with open(zutil_path, 'w') as f:
                    f.write(content)
                print("[*] Patched zutil.h for MacOS fdopen conflict.")
            else:
                print("[!] zlib/zutil.h not found, skipping patch.")
            print(f"[+] GCC source acquired at {GCC_SOURCE_DIR}.")
        else:
             print("[*] Confirmed GCC source is available.")
        # --- 5. Build Components in Order ---
        build_steps: List[dict] = [
            {"name": "gmp-6.3.0", "dir": os.path.join(N64SDK_DIR, "gmp-6.3.0"),
             "configure_cmd": ["./configure", "--prefix=" + N64_INST], "has_configure": True},
            {"name": "mpfr-4.2.2", "dir": os.path.join(N64SDK_DIR, "mpfr-4.2.2"),
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--with-gmp=" + N64_INST], "has_configure": True},
            {"name": "mpc-1.3.1", "dir": os.path.join(N64SDK_DIR, "mpc-1.3.1"),
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--with-gmp=" + N64_INST, "--with-mpfr=" + N64_INST, "--disable-shared"], "has_configure": True},
            {"name": "binutils-gdb", "dir": BINUTILS_DIR,
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--target=mips64-elf", "--disable-werror", "--with-gmp=" + N64_INST, "--with-mpfr=" + N64_INST, "--disable-gdb", "--disable-gprofng"],
             "env": {**os.environ, "CC_FOR_BUILD": "clang", "CFLAGS": "-Wno-error -Wno-deprecated-non-prototype", "CXXFLAGS": "-Wno-error -Wno-deprecated-non-prototype", "WERROR_CFLAGS": "", "WARN_CFLAGS": "", "gl_cv_warn_cflags": ""}, "has_configure": True},
            # --- Stage 1 GCC (ISL REMOVED FROM CONFIGURE) ---
            {"name": "gcc-stage1", "dir": GCC_SOURCE_DIR,
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--target=mips64-elf", "--enable-languages=c", "--with-gmp=" + N64_INST, "--with-mpfr=" + N64_INST, "--with-mpc=" + N64_INST, "--disable-multilib", "--disable-libssp", "--disable-libstdcxx", "--disable-libquadmath", "--disable-nls", "--without-headers", "--disable-shared", "--disable-threads", "--disable-werror", "--with-abi=n64", "--with-system-zlib"],
             "env": {**os.environ, "CC_FOR_BUILD": "clang", "CFLAGS": "-Wno-error", "CXXFLAGS": "-Wno-error", "LDFLAGS": f"-L{os.path.join(N64_INST, 'lib')}", "CPPFLAGS": f"-I{os.path.join(N64_INST, 'include')}"}, "has_configure": True},
            # --- Newlib ---
            {"name": "newlib-cygwin-3.6.4", "dir": os.path.join(N64SDK_DIR, "newlib-cygwin-3.6.4"),
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--target=mips64-elf"],
             "env": {**os.environ, "PATH": f"{os.path.join(N64_INST, 'bin')}:{os.environ.get('PATH', '')}"}, "has_configure": True},
            # --- Stage 2 GCC (ISL REMOVED FROM CONFIGURE) ---
            {"name": "gcc-stage2", "dir": GCC_SOURCE_DIR,
             "configure_cmd": ["./configure", "--prefix=" + N64_INST, "--target=mips64-elf", "--enable-languages=c,c++", "--with-gmp=" + N64_INST, "--with-mpfr=" + N64_INST, "--with-mpc=" + N64_INST, "--disable-multilib", "--with-newlib", "--disable-werror", "--with-abi=n64", "--with-system-zlib"],
             "env": {**os.environ, "CC_FOR_BUILD": "clang", "CFLAGS": "-Wno-error", "CXXFLAGS": "-Wno-error", "LDFLAGS": f"-L{os.path.join(N64_INST, 'lib')}", "CPPFLAGS": f"-I{os.path.join(N64_INST, 'include')}"}, "has_configure": True},
            {"name": "libdragon-toolchain-continuous-prerelease", "dir": os.path.join(N64SDK_DIR, "libdragon-toolchain-continuous-prerelease"),
             "configure_cmd": [], "has_configure": False, "make_install_cmd": ["make", "install", f"INSTALL_PATH={N64_INST}"]}
        ]
        for step in build_steps:
            component_name = step["name"]
            build_dir = step["dir"]
            has_configure = step["has_configure"]
            if is_build_complete(component_name, build_dir, N64_INST):
                print(f"[*] Build for '{component_name}' appears complete. Skipping.")
                continue
            if not os.path.exists(build_dir):
                print(f"[!] Build directory for {component_name} not found. Skipping.")
                continue
            os.chdir(build_dir)
            print(f"[*] Building {component_name} in {build_dir}...")
            try:
                if has_configure:
                    configure_cmd = step["configure_cmd"]
                    configure_script_path = "./configure"
                    if os.path.exists(configure_script_path):
                        st = os.stat(configure_script_path)
                        os.chmod(configure_script_path, st.st_mode | stat.S_IEXEC)
                        print(f"[*] Made '{configure_script_path}' executable.")
                    print(f"[CMD] {' '.join(configure_cmd)}")
                    env_to_use = step.get("env", os.environ)
                    subprocess.check_call(configure_cmd, env=env_to_use)
                make_cmd = ["make"]
                print(f"[CMD] {' '.join(make_cmd)}")
                permission_fix_applied = False
                if component_name == "gmp-6.3.0":
                    m4_ccas_path = os.path.join(build_dir, "mpn", "m4-ccas")
                    if os.path.exists(m4_ccas_path):
                        try:
                            st = os.stat(m4_ccas_path)
                            os.chmod(m4_ccas_path, st.st_mode | stat.S_IEXEC)
                            print(f"[*] Made '{m4_ccas_path}' executable for GMP.")
                            permission_fix_applied = True
                        except Exception as e:
                            print(f"[!] Warning making '{m4_ccas_path}' executable: {e}")
                elif component_name == "mpfr-4.2.2":
                    get_patches_sh_path = os.path.join(build_dir, "tools", "get_patches.sh")
                    if os.path.exists(get_patches_sh_path):
                        try:
                            st = os.stat(get_patches_sh_path)
                            os.chmod(get_patches_sh_path, st.st_mode | stat.S_IEXEC)
                            print(f"[*] Made '{get_patches_sh_path}' executable for MPFR.")
                            permission_fix_applied = True
                        except Exception as e:
                            print(f"[!] Warning making '{get_patches_sh_path}' executable: {e}")
                if permission_fix_applied:
                    print(f"[*] Applied pre-build permission fixes for {component_name}.")
                env_to_use_make = step.get("env", os.environ)
                subprocess.check_call(make_cmd, env=env_to_use_make)
                make_install_cmd = step.get("make_install_cmd", ["make", "install"])
                print(f"[CMD] {' '.join(make_install_cmd)}")
                subprocess.check_call(make_install_cmd, env=env_to_use_make)
                print(f"[+] {component_name} built and installed successfully.")
                # Verify after installation
                if not verify_install(component_name, build_dir, N64_INST):
                    return False
                # --- Enhanced gcc-stage1 check ---
                if component_name == "gcc-stage1":
                    gcc_stage1_bin = os.path.join(N64_INST, "bin", "mips64-elf-gcc")
                    if not os.path.isfile(gcc_stage1_bin):
                        print(f"[!] gcc-stage1 completed but '{gcc_stage1_bin}' missing.")
                        return False
                    print(f"[*] Testing '{gcc_stage1_bin}'...")
                    test_c = "int main(){return 0;}"
                    with open("test.c", "w") as f: f.write(test_c)
                    try:
                        test_cmd = [gcc_stage1_bin, "test.c", "-o", "test.exe"]
                        print(f"[CMD] {' '.join(test_cmd)}")
                        subprocess.check_call(test_cmd)
                        print("[*] gcc-stage1 compiler test successful.")
                    except subprocess.CalledProcessError as e:
                        print(f"[!] gcc-stage1 compiler test failed: {e}")
                        return False
                    finally:
                        for f in ["test.c", "test.exe"]: # Clean up even if test fails
                            if os.path.exists(f): os.remove(f)
            except subprocess.CalledProcessError as e:
                print(f"[!] Error building {component_name}: {e}")
                with open(os.path.join(WORK, f"{component_name}_build_log.txt"), "w") as log_file:
                    log_file.write(str(e) + "\n" + (e.output.decode() if e.output else ""))
                print(f"[!] Detailed logs saved to {component_name}_build_log.txt")
                if component_name in ["gcc-stage1", "binutils-gdb"]: # Stop on critical errors
                    print(f"[!] Critical build failure for {component_name}. Stopping.")
                    return False
            except Exception as e:
                print(f"[!] Unexpected error building {component_name}: {e}")
                traceback.print_exc()
                with open(os.path.join(WORK, f"{component_name}_build_log.txt"), "w") as log_file:
                    log_file.write(traceback.format_exc())
                print(f"[!] Detailed logs saved to {component_name}_build_log.txt")
                return False
            finally:
                os.chdir(WORK)
        print("[+] N64 development environment setup complete.")
        print(f"    SDK components in: {N64SDK_DIR}")
        print(f"    Toolchain installed to: {N64_INST}")
        print(f"    Add '{os.path.join(N64_INST, 'bin')}' to PATH.")
        return True
    # --- Compilation and Running Functions ---
    # (Assuming these functions exist in your full script)
    def compile_c_project(source_path: str, output_name: str) -> Optional[str]:
        """Compiles a C file to a Z64 ROM using the built Libdragon."""
        # Implementation would go here...
        print(f"[!] compile_c_project not fully implemented in this snippet for brevity.")
        return None
    def compile_hello_world():
        """Creates and compiles a Hello World ROM."""
        print("[!] compile_hello_world not fully implemented in this snippet for brevity.")
        input("[Press Enter to continue]")
    def compile_custom_c():
        """Compile user-provided C file."""
        print("[!] compile_custom_c not fully implemented in this snippet for brevity.")
        input("[Press Enter to continue]")
    def compile_rom_func():
        """Compile a dummy ROM (Placeholder)."""
        print("[!] compile_rom_func not fully implemented in this snippet for brevity.")
        input("[Press Enter to continue]")
    def run_rom(rom_path_override: Optional[str] = None, use_dialog: bool = False):
        """Launches N64 Emulation using mupen64plus."""
        print("[!] run_rom not fully implemented in this snippet for brevity.")
        input("[Press Enter to continue]")








    # --- Menu System ---
    def menu_main():
        """Main loop for the N64 Agent menu."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\

┌─────────────────────────────────────────────┐
│ N64Dev Agent (Custom n64dev)        │
│ Workspace: {WORK}                           │
├─────────────────────────────────────────────┤
│ 1) Setup N64 tool-chain (Custom Build)      │
│ 2) Compile Hello-World (C)                  │
│ 3) Compile Custom C Project                 │
│ 4) Compile Dummy ROM (Placeholder)          │
│ 5) Run ROM in mupen64plus                   │
│ 6) Run Custom ROM (Dialog)                  │
│ 7) Exit to Main Cyber Agent Menu            │
└─────────────────────────────────────────────┘
""")
            choice = input("Select [1-7]: ").strip()
            if choice == "1":
                success = setup_chain()
                if success:
                    print("[*] Toolchain setup completed successfully.")
                else:
                    print("[!] Toolchain setup failed.")
                input("[Press Enter to continue]")
            elif choice == "2":
                compile_hello_world()
            elif choice == "3":
                compile_custom_c()
            elif choice == "4":
                compile_rom_func()
            elif choice == "5":
                run_rom()
            elif choice == "6":
                run_rom(use_dialog=True)
            elif choice == "7":
                print("[*] Returning to main Cyber Agent menu.")
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Entry Point Logic ---
    print("[*] Starting N64 Development Agent (Custom n64dev Build)...")
    print(f"[*] N64 Dev Agent workspace: {WORK}")
    
    # Handle direct ROM run from argument (like command line)
    if rom_path_override:
        rom_override_path = Path(rom_path_override)
        if rom_override_path.is_file() and rom_override_path.suffix.lower() == '.z64':
            print(f"[*] Running ROM provided via argument: {rom_override_path}")
            run_rom(rom_path_override=str(rom_override_path))
            # After running, still show the menu unless you want to exit completely
            # For now, let's show the menu after the direct run.
        else:
            print(f"[!] Invalid ROM path provided via argument: {rom_path_override}")
            # Show menu to allow user to correct or choose another option
    # Always show the main menu after initialization and potential direct run
    menu_main()
    
    print("[*] N64 Development Agent finished. Returning to main menu.")
    os.chdir(ORIGINAL_CWD_AGENT)
    return

# - End of n64dev_agent function -













































































































#######################################################################################################
# 5. Ghidra Agent
#######################################################################################################
def launch_ghidra_agent():
    # Ghidra Agent code is encapsulated within ghidra_agent function (as provided)
    # We just call it here.
    # Note: This function calls sys.exit(0) upon completion, so execution won't return here normally.
    ghidra_agent()

#######################################################################################################
# 6. Cydia Module Manager
#######################################################################################################
def launch_cydia_agent():
    # Cydia Agent code is encapsulated within cydia_agent function (as provided)
    # We just call it here.
    # Note: This function calls sys.exit(0) upon completion.
    cydia_agent()

# --- LLM Backend Selection and Loading ---
def select_and_load_model():
    global BACKEND, MODEL, MODEL_PATH, MODEL_DIR

    # Load Cydia dynamic tools before selecting any model/agent
    load_cydia_dynamic_tools()

    print("Choose model:")
    print("1. Gemini 1.5 Pro (.gguf via llama.cpp)")
    print("2. Deepseek R1 7B (via ollama)")
    print("3. Qwen (.safetensors via transformers)")
    print("4. N64 Dev Agent")
    print("5. Ghidra Agent")
    print("6. Cydia Module Manager")

    choice = input("Enter choice (1/2/3/4/5/6): ").strip()
    BACKEND = None

    # Simplified main menu logic calling dedicated functions
    if choice == "1":
        load_gemini_model()
    elif choice == "2":
        load_deepseek_model()
    elif choice == "3":
        load_qwen_model()
    elif choice == "4":
        launch_n64_agent()
        # If n64 agent doesn't exit, you might need specific logic here
    elif choice == "5":
        launch_ghidra_agent()
        # If ghidra agent doesn't exit, you might need specific logic here
    elif choice == "6":
        launch_cydia_agent()
        # If cydia agent doesn't exit, you might need specific logic here
    else:
        print("Invalid choice.")
        sys.exit(1)

# - LLM Function -
def llm(prompt: str) -> str:
    if BACKEND == "llama":
        response = MODEL(prompt=prompt, max_tokens=2000, temperature=0.7, top_p=0.95, stop=[""])  # Common stop sequence
        return response['choices'][0]['text']
    elif BACKEND == "ollama":
        import requests
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama API: {e}"
    elif BACKEND == "transformers":
        # Updated to use the new MODEL structure for transformers
        try:
            if isinstance(MODEL, dict) and MODEL.get("type") == "transformers":
                tokenizer = MODEL["tokenizer"]
                model = MODEL["model"]
                device = model.device # Get the device the model is on

                # Use the recommended Qwen chat template method
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512, # Adjust as needed
                    # Add other generation parameters if desired (e.g., temperature, top_p)
                )
                # Remove input tokens from generated output
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
            else:
                return "Error: Transformers MODEL not loaded correctly or unexpected format."
        except Exception as e:
            return f"Error using Transformers model: {e}"
    else:
        return "LLM backend not initialized correctly."

# --- Agent Reasoning and Parsing ---
def agent_reason(query: str) -> str:
    """Main agent reasoning loop using ReAct."""
    # Include dynamic tools in the prompt
    tools_desc = "\n".join([f"- {k}" for k in tool_dict.keys()])
    prompt = f"""
You are a helpful AI assistant with access to various tools. Use them to answer the user's request.
Available tools:
{tools_desc}
Command formats:
- Archive: archive -d [dir] -o [output]
- SSH: do_shelly [ip] [command]
- SEO: seo_analyze_site [url]
- GUI: recognize_gui_elements [description]
Special shortcuts:
- If query starts with 'site analyze:', treat as seo_analyze_site with the URL after ':'.
Output in ReAct format:
Thought: [reasoning]
Action: [tool_name]
Action Input: [arguments in JSON format]
Observation: [result of the action]
... (repeat Thought/Action/Action Input/Observation as needed) ...
Final Answer: [final response to the user]
Begin!
Query: {query}
"""
    try:
        response = llm(prompt)
        return response
    except Exception as e:
        return f"Error in agent reasoning: {e}"

def parse_agent_output(output: str) -> Tuple[str, str]:
    """Parse the output from the agent to extract action and input."""
    # Use regex to find Action and Action Input
    action_match = re.search(r"Action:\s*(\w+)", output)
    input_match = re.search(r"Action Input:\s*(.*)", output, re.DOTALL)
    if action_match and input_match:
        action = action_match.group(1).strip()
        input_str = input_match.group(1).strip()
        # Try to parse the input string as JSON
        try:
            input_args = json.loads(input_str)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a string argument
            input_args = input_str
        return action, input_args
    else:
        # If regex doesn't match, it might be a malformed action or just text
        # You could log this or handle it differently
        pass
    return "continue", output

# - Chat and GUI Modes -
def agent_chat() -> None:
    """Main chat interface for the agent."""
    print("=== Cyber Agent Chat (Integrated Archive, SSH, SEO, GUI) ===")
    print("(Type 'show tools' to see available actions, 'reload tools' to reload from Cydia)")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            # --- NEW: Handle special commands for agent interaction ---
            if q.lower() == "show tools":
                print("\n--- Available Actions/Tools ---")
                for tool_name in sorted(tool_dict.keys()):
                    print(f"  {tool_name}")
                print("-------------------------------")
                continue
            elif q.lower() == "reload tools":
                print("[*] Reloading dynamic tools from Cydia modules...")
                load_cydia_dynamic_tools()
                print("[+] Tools reloaded.")
                continue
            # --- END NEW ---
            elif q.lower().startswith("site analyze:"):
                url = q[len("site analyze:"):].strip()
                result = seo_analyze_site(url)
                print(result)
                continue
            response = agent_reason(q)
            # print(f"[DEBUG] LLM Response:\n{response}") # Uncomment for debugging
            # Simple parsing loop
            max_iterations = 5
            iterations = 0
            current_output = response
            while iterations < max_iterations:
                action, args = parse_agent_output(current_output)
                if action == "continue":
                    print(current_output)
                    break
                elif action in tool_dict:
                    print(f"[Action] {action}({args})")
                    try:
                        observation = tool_dict[action](args)
                        print(f"[Observation] {observation}")
                        follow_up_prompt = f"{response}\nObservation: {observation}\n"
                        current_output = llm(follow_up_prompt)
                        # print(f"[DEBUG] LLM Follow-up Response:\n{current_output}") # Uncomment for debugging
                    except Exception as e:
                        error_msg = f"[Error] Tool '{action}' failed: {e}"
                        print(error_msg)
                        follow_up_prompt = f"{response}\nObservation: {error_msg}\n"
                        current_output = llm(follow_up_prompt)
                else:
                    print(f"[Error] Unknown action: {action}")
                    break
                iterations += 1
            else:
                print("[Error] Maximum iterations reached. Stopping.")

# --- Full Agent Code from Uploaded File ---

# --- N64 Dev Agent (Full Code) ---
# (This is the full n64dev_agent function from your uploaded file)
def n64dev_agent():
    import os
    import sys
    import subprocess
    import shutil
    import struct
    import tempfile
    import pathlib
    from typing import Optional

    # --- Configuration ---
    # Paths (Hardcoded - adjust as needed)
    # Consider making these configurable via environment variables or a config file
    MIPS_GCC_PATH = "/opt/cross/mips64-linux-gnu/bin/mips64-linux-gnu-gcc" # Example path
    MIPS_LD_PATH = "/opt/cross/mips64-linux-gnu/bin/mips64-linux-gnu-ld"   # Example path
    QEMU_MIPS_PATH = "/opt/homebrew/bin/qemu-mips" # Example path for macOS (adjust for Linux)
    MUPEN64PLUS_PATH = "mupen64plus" # Assume it's in PATH or adjust
    PYN64_PATH = "/Users/rosty/Desktop/PyN64/src" # Example path to PyN64 source
    # Ensure PyN64 is in the path for import
    sys.path.insert(0, PYN64_PATH)

    # ROM paths (Example defaults, can be overridden)
    DEFAULT_ROM_PATH = "/Users/rosty/Desktop/N64_ROMS/default.n64" # Example default ROM
    CUSTOM_ROM_OUTPUT_DIR = "/Users/rosty/Desktop/N64_ROMS/Custom" # Directory for custom ROMs
    os.makedirs(CUSTOM_ROM_OUTPUT_DIR, exist_ok=True)

    # --- Global State ---
    # Store the path of the last successfully compiled custom ROM
    last_custom_compiled_rom = None

    # --- Utility Functions ---
    def log(message: str):
        """Simple logging function."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def check_tool(tool_path: str, tool_name: str) -> bool:
        """Check if a tool exists."""
        if not os.path.isfile(tool_path) or not os.access(tool_path, os.X_OK):
            log(f"Error: {tool_name} not found or not executable at {tool_path}")
            return False
        return True

    def check_command_in_path(command: str) -> bool:
        """Check if a command exists in the system PATH."""
        return shutil.which(command) is not None

    def get_menu_choice() -> str:
        """Display menu and get user choice."""
        print("\n--- N64 Development Agent ---")
        print("1. Setup N64 Toolchain (Stub)")
        print("2. Compile Hello World ROM")
        print("2b. Compile Custom C File")
        print("3. Run Last Custom ROM")
        print("3b. Run ROM (Select File)")
        print("4. Back to Main Menu")
        choice = input("Choose an option (1/2/2b/3/3b/4): ").strip().lower()
        return choice

    def setup_chain():
        """Placeholder for toolchain setup."""
        print("Setting up N64 toolchain... (Stub)")
        # This would involve downloading and compiling the toolchain
        # (e.g., using libdragon or naken_asm or a custom GCC setup)
        # For now, we assume the tools are pre-installed at MIPS_GCC_PATH etc.
        # You could add checks here for the required binaries.
        # Example check:
        # if not check_tool(MIPS_GCC_PATH, "MIPS GCC"):
        #     print("Please ensure MIPS GCC is installed and path is correct.")
        #     return
        print("Toolchain setup check completed (stub).")

    def compile_rom_func():
        """Compile a simple Hello World ROM."""
        # Simple Hello World C code for N64 (requires libdragon or similar)
        hello_c_code = '''
    #include <stdio.h>
    #include <malloc.h>
    #include <string.h>
    #include <stdint.h>
    #include <libdragon.h>

    int main(void)
    {
        // Initialize peripherals
        console_init();
        console_set_render_mode(RENDER_MANUAL);

        while (1) {
            console_clear();
            printf("Hello, N64 World!\\n");
            printf("Press A to continue.\\n");

            // Update the console display
            console_render();
        }

        return 0;
    }
    '''
        # Create a temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = os.path.join(tmpdir, "hello.c")
            elf_file = os.path.join(tmpdir, "hello.elf")
            rom_file = os.path.join(CUSTOM_ROM_OUTPUT_DIR, "hello_world.n64")

            # Write C code to temp file
            with open(src_file, 'w') as f:
                f.write(hello_c_code)

            # --- Compilation Steps ---
            # 1. Compile C to object file (requires libdragon headers/libs)
            # This is a simplified example command. Real setup requires proper flags and includes.
            # Example using libdragon:
            # gcc -std=gnu99 -march=vr4300 -mtune=vr4300 -O2 -I/opt/libdragon/include -L/opt/libdragon/lib -ldragon -lc -lm -ldragonsys -o hello.elf hello.c
            # For this stub, we'll just create a dummy ELF file.
            log(f"Compiling {src_file} to {elf_file}... (Stub)")
            # In a real scenario, you'd run the MIPS GCC command here.
            # subprocess.run([MIPS_GCC_PATH, ...], check=True, cwd=tmpdir)
            # For now, create a dummy file to simulate success.
            with open(elf_file, 'wb') as f:
                f.write(b"DUMMY_ELF_CONTENT_FOR_HELLO_WORLD")

            # 2. Convert ELF to ROM (requires a tool like 'elf2n64' or linker script)
            # This step is highly dependent on your toolchain setup.
            # For this stub, we'll just copy the dummy ELF to a .n64 file.
            log(f"Converting {elf_file} to {rom_file}... (Stub)")
            shutil.copyfile(elf_file, rom_file)

            log(f"Hello World ROM compiled successfully: {rom_file}")

    def compile_custom_c_file():
        """Compile a user-provided C file into an N64 ROM."""
        global last_custom_compiled_rom
        c_file_path = input("Enter the path to your .c file: ").strip()
        if not os.path.isfile(c_file_path):
            print("Error: File not found.")
            return

        rom_name = input("Enter the desired ROM name (without .n64 extension): ").strip()
        if not rom_name:
            rom_name = "custom_rom"
        rom_file_name = f"{rom_name}.n64"
        rom_output_path = os.path.join(CUSTOM_ROM_OUTPUT_DIR, rom_file_name)

        # Create a temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy user's C file to temp dir (could be multiple files)
            temp_c_file = os.path.join(tmpdir, os.path.basename(c_file_path))
            shutil.copyfile(c_file_path, temp_c_file)
            elf_file = os.path.join(tmpdir, f"{rom_name}.elf")

            # --- Compilation Steps ---
            # 1. Compile C to ELF (requires N64 toolchain setup)
            log(f"Compiling {temp_c_file} to {elf_file}... (Stub)")
            # In a real implementation, you would call MIPS GCC here with the correct flags.
            # Example (if libdragon is used):
            # cmd = [
            #     MIPS_GCC_PATH,
            #     "-std=gnu99", "-march=vr4300", "-mtune=vr4300", "-O2",
            #     "-I/opt/libdragon/include", "-L/opt/libdragon/lib",
            #     "-ldragon", "-lc", "-lm", "-ldragonsys",
            #     "-o", elf_file, temp_c_file
            # ]
            # subprocess.run(cmd, check=True, cwd=tmpdir)
            # For this stub, create a dummy ELF.
            with open(elf_file, 'wb') as f:
                f.write(f"DUMMY_ELF_CONTENT_FOR_{rom_name}".encode())

            # 2. Convert ELF to ROM
            log(f"Converting {elf_file} to {rom_output_path}... (Stub)")
            # This requires a specific tool like 'elf2n64' or a custom linker script.
            # For this stub, just copy the dummy ELF.
            shutil.copyfile(elf_file, rom_output_path)

        log(f"Custom ROM compiled successfully: {rom_output_path}")
        last_custom_compiled_rom = rom_output_path # Store the path

    def run_rom(rom_path: str):
        """Attempt to run an N64 ROM using available emulators."""
        if not os.path.isfile(rom_path):
            print(f"Error: ROM file not found: {rom_path}")
            return

        print(f"Attempting to run ROM: {rom_path}")

        # --- Emulator Selection Logic ---
        # 1. Try PyN64 (if available and working)
        try:
            import n64cpu # Import from PyN64
            print("Attempting to run with PyN64...")
            # PyN64 typically requires more setup (loading PIF, etc.)
            # This is a simplified call and might not work out-of-the-box.
            # pif_path = "/path/to/pifdata.bin" # Required for PyN64
            # n64 = n64cpu.N64(pif_path=pif_path, rom_path=rom_path)
            # n64.run()
            print("PyN64 execution logic goes here (stub). Requires PIF ROM setup.")
            # If PyN64 runs successfully, return
            # return
        except ImportError:
            print("PyN64 not found or import failed.")
        except Exception as e:
            print(f"PyN64 failed to run ROM: {e}")

        # 2. Try QEMU (user-mode, less common for full N64 emulation)
        if check_command_in_path("qemu-mips"):
            print("Attempting to run with QEMU (MIPS user-mode - limited N64 support)...")
            # QEMU user-mode emulation for N64 ROMs is non-trivial and usually not done this way.
            # It would require significant setup and likely wouldn't run a full game.
            # This is just a placeholder check.
            print("QEMU execution for N64 ROMs is complex and not directly supported like this (stub).")
        else:
            print("QEMU (MIPS) not found in PATH.")

        # 3. Try Mupen64Plus (most likely to work)
        if check_command_in_path("mupen64plus"):
            print("Attempting to run with Mupen64Plus...")
            try:
                # Run Mupen64Plus with the ROM
                subprocess.run([MUPEN64PLUS_PATH, rom_path], check=True)
                print("ROM execution finished (Mupen64Plus).")
                return # Success
            except subprocess.CalledProcessError as e:
                print(f"Mupen64Plus failed to run ROM (exit code {e.returncode}).")
            except Exception as e:
                print(f"Error running Mupen64Plus: {e}")
        else:
            print("Mupen64Plus not found in PATH.")

        # If all methods fail
        print("Failed to run ROM. Please ensure an N64 emulator (like Mupen64Plus) is installed and configured.")

    def menu_main():
        """Main loop for the N64 Agent menu."""
        while True:
            choice = get_menu_choice()
            if choice == "1":
                setup_chain()
            elif choice == "2":
                compile_rom_func() # Existing Hello World compile
            elif choice == "2b": # <<< Handle NEW Compile Option >>>
                compile_custom_c_file() # Call the new function
            elif choice == "3": # <<< Run ROM (potentially last compiled or default) >>>
                last_custom_rom = getattr(n64dev_agent, 'last_custom_compiled_rom', None)
                if last_custom_rom and os.path.isfile(last_custom_rom):
                    print(f"[*] Running last compiled custom ROM: {last_custom_rom}")
                    run_rom(last_custom_rom)
                else:
                    print("[!] No last compiled custom ROM found. Running default ROM.")
                    run_rom(DEFAULT_ROM_PATH)
            elif choice == "3b": # <<< Run ROM (Select File) >>>
                # Simple file selection (can be improved with a file dialog library)
                rom_path_override = input("Enter the full path to the ROM file (or press Enter for default): ").strip()
                if not rom_path_override:
                    rom_path_override = DEFAULT_ROM_PATH
                run_rom(rom_path_override)
            elif choice == "4": # <<< Exit to Main Menu >>>
                print("Returning to main Cyber Agent menu.")
                break # Exit the N64 agent loop
            else:
                print("Invalid choice.")

    # --- Entry Point for N64 Agent ---
    print("Starting N64 Development Agent...")
    # Initial setup check
    setup_chain()
    # Start the menu loop
    menu_main()
    # Signal that the agent has finished its task
    sys.exit(0) # Exit the entire script when N64 agent is done

# --- Ghidra Agent (Full Code) ---
# (This is the full ghidra_agent function from your uploaded file)
def ghidra_agent():
    import os
    import sys
    import subprocess
    import shutil
    import struct
    import json
    import tempfile
    import pathlib
    from typing import Optional, Dict, Any

    # --- Configuration (Hardcoded - adjust as needed) ---
    # Consider making these configurable via environment variables or a config file
    GHIDRA_PATH = "/Applications/ghidra_11.0.1_PUBLIC"  # Adjust this path to your Ghidra installation
    GHIDRA_HEADLESS_PATH = os.path.join(GHIDRA_PATH, "support", "analyzeHeadless")
    GHIDRA_OUTPUT_DIR = "/Users/rosty/Desktop/Ghidra_Output"  # Default output directory for Ghidra results
    os.makedirs(GHIDRA_OUTPUT_DIR, exist_ok=True)  # Ensure output dir exists

    # --- Global State for Ghidra Agent ---
    CURRENT_ROM_PATH: Optional[str] = None
    CURRENT_PROJECT_NAME: str = "TempGhidraProject"
    CURRENT_PROJECT_PATH: str = tempfile.mkdtemp(prefix="ghidra_proj_")
    ACTION_HISTORY: list = []  # To store recent actions

    # --- Utility Functions ---
    def log(message: str):
        """Simple logging function."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def store(action_data: Dict[str, Any]):
        """Store action data to history."""
        global ACTION_HISTORY
        ACTION_HISTORY.append(action_data)
        # Keep history to last 10 actions
        ACTION_HISTORY = ACTION_HISTORY[-10:]

    def load_rom(rom_path: str) -> bool:
        """Load a ROM file for analysis."""
        global CURRENT_ROM_PATH
        rom_path_obj = pathlib.Path(rom_path)
        if not rom_path_obj.is_file():
            log(f"Error: ROM file not found: {rom_path}")
            return False
        CURRENT_ROM_PATH = str(rom_path_obj.resolve())
        log(f"ROM loaded: {CURRENT_ROM_PATH}")
        store({"action": "load_rom", "rom_path": CURRENT_ROM_PATH})
        return True

    def run_headless_analysis(project_path: str, project_name: str, import_file_path: str, pre_scripts=None, post_scripts=None) -> bool:
        """Run Ghidra headless analysis."""
        if not os.path.isfile(GHIDRA_HEADLESS_PATH):
            log(f"Error: Ghidra analyzeHeadless script not found at {GHIDRA_HEADLESS_PATH}")
            return False

        cmd = [GHIDRA_HEADLESS_PATH, project_path, project_name, "-import", import_file_path, "-overwrite", "-noanalysis"]

        if pre_scripts:
            for script in pre_scripts:
                cmd.extend(["-preScript", script])
        if post_scripts:
            for script in post_scripts:
                cmd.extend(["-postScript", script])

        cmd.append("-deleteProject") # Delete the temporary project afterwards

        try:
            log(f"Running Ghidra headless analysis: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log("Ghidra headless analysis completed successfully.")
            # print(f"[DEBUG] Ghidra stdout:\n{result.stdout}") # Uncomment for debugging
            # print(f"[DEBUG] Ghidra stderr:\n{result.stderr}") # Uncomment for debugging
            return True
        except subprocess.CalledProcessError as e:
            log(f"Ghidra headless analysis failed (return code {e.returncode}).")
            log(f"Stdout: {e.stdout}")
            log(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            log(f"Error running Ghidra headless analysis: {e}")
            return False

    def pseudo_disassemble(rom_path: str, output_dir: str) -> str:
        """Run simulated AI disassembly to pseudo-assembly."""
        rom_path_obj = pathlib.Path(rom_path)
        if not rom_path_obj.is_file():
            return f"Error: ROM file not found: {rom_path}"

        output_file = pathlib.Path(output_dir) / f"{rom_path_obj.stem}_disasm.txt"
        log(f"Starting simulated AI disassembly for {rom_path}...")

        try:
            with open(rom_path, 'rb') as f:
                rom_bytes = f.read()

            # --- AI Heuristic Analysis ---
            header_info = f"; Simulated Disassembly for {rom_path_obj.name}\n"
            header_info += f"; File Size: {len(rom_bytes)} bytes\n"
            header_info += "; --- AI Heuristics Applied ---\n"

            # 1. Find potential entry point (e.g., look for common MIPS start sequences)
            entry_point_offset = None
            mips_start_sequences = [b'\x80\x37\x12\x40', b'\x3C\x08\x80\x00', b'\x3C\x1A\x80\x00'] # Examples
            for seq in mips_start_sequences:
                pos = rom_bytes.find(seq)
                if pos != -1:
                    entry_point_offset = pos
                    header_info += f"; AI Heuristic: Found potential MIPS start sequence '{seq.hex()}' at offset 0x{pos:08X}\n"
                    break
            if entry_point_offset is None:
                entry_point_offset = 0x1000 # Default for N64?
                header_info += f"; AI Heuristic: No clear start sequence, assuming default entry at 0x{entry_point_offset:08X}\n"

            # 2. Disassemble a small snippet around the entry point
            snippet_start = max(0, entry_point_offset - 0x20)
            snippet_end = min(len(rom_bytes), entry_point_offset + 0x100)
            snippet_bytes = rom_bytes[snippet_start:snippet_end]

            disasm_snippet = f"\n; --- Disassembly Snippet (around 0x{entry_point_offset:08X}) ---\n"
            for i in range(0, len(snippet_bytes), 4):
                word_offset = snippet_start + i
                if word_offset < len(rom_bytes) - 3:
                    word_bytes = rom_bytes[word_offset:word_offset+4]
                    word_int = struct.unpack('>I', word_bytes)[0] # Big-endian MIPS
                    disasm_snippet += f"{word_offset:08X}: {word_bytes.hex().upper()} ; Simulated disasm: lui $t0, 0x{word_int >> 16:04X}\n" # Placeholder

            # 3. Find potential strings (ASCII sequences)
            strings_found = []
            min_str_len = 4
            # Simple search for ASCII strings
            try:
                current_str = b""
                current_start = 0
                for i, byte in enumerate(rom_bytes):
                    if 32 <= byte <= 126: # Printable ASCII
                        if not current_str:
                            current_start = i
                        current_str += bytes([byte])
                    else:
                        if len(current_str) >= min_str_len:
                            strings_found.append((current_start, current_str))
                        current_str = b""
                # Catch string at the end
                if len(current_str) >= min_str_len:
                     strings_found.append((current_start, current_str))

                # Deduplicate and filter strings
                unique_strings = list(set(s[1] for s in strings_found))
                if unique_strings:
                    strings_found.clear() # Clear offsets, keep unique content
                    strings_found.append("; - Potential Strings Found (AI Heuristic) -")
                    for s in unique_strings[:20]: # Limit output
                        try:
                            decoded_str = s.decode('ascii')
                            strings_found.append(f" \"...{decoded_str}...\"")
                        except UnicodeDecodeError:
                            pass
            except Exception as e:
                log(f"Warning: Error during string search: {e}")

            # Combine output
            full_output = header_info + disasm_snippet + "\n".join(strings_found) + "\n; - End of Simulated Disassembly -\n"

            with open(output_file, 'w') as f:
                f.write(full_output)

            log(f"Simulated disassembly saved to {output_file}")
            store({
                "action": "pseudo_disassemble",
                "rom_path": rom_path,
                "output_path": str(output_file),
                "entry_point": f"0x{entry_point_offset:08X}",
                "strings_found": len(strings_found) - 1 if strings_found else 0 # Adjust count
            })
            return f"Simulated disassembly complete. Output saved to: {output_file}\nKey findings:\n- Entry point heuristic: 0x{entry_point_offset:08X}\n- Strings found: {len(strings_found) - 1 if strings_found else 0}"

        except Exception as e:
            log(f"Error during simulated disassembly: {e}")
            return f"Error during simulated disassembly: {e}"

    def pseudo_decompile(rom_path: str, output_dir: str) -> str:
        """Run simulated AI decompilation to pseudo-C."""
        rom_path_obj = pathlib.Path(rom_path)
        if not rom_path_obj.is_file():
            return f"Error: ROM file not found: {rom_path}"

        output_file = pathlib.Path(output_dir) / f"{rom_path_obj.stem}_decomp.c"
        log(f"Starting simulated AI decompilation for {rom_path}...")

        try:
            with open(rom_path, 'rb') as f:
                rom_bytes = f.read()

            # --- AI Heuristic Analysis for Decompilation ---
            header_comment = f"/*\n * Simulated Decompile for {rom_path_obj.name}\n * File Size: {len(rom_bytes)} bytes\n */\n\n"

            # 1. Find potential 'main' function prologue (e.g., stack setup)
            # Very simplified heuristic for MIPS N64
            main_func_offset = None
            # Example pattern: addiu $sp, $sp, -X (stack allocation)
            # This is highly unreliable without real analysis
            for i in range(0, min(0x10000, len(rom_bytes) - 4), 4): # Search first 64KB
                word_bytes = rom_bytes[i:i+4]
                word_int = struct.unpack('>I', word_bytes)[0]
                opcode = word_int >> 26
                if opcode == 0b001001: # MIPS addiu
                    rt = (word_int >> 16) & 0x1F
                    rs = (word_int >> 21) & 0x1F
                    immediate = word_int & 0xFFFF
                    # Check if it's 'addiu $sp, $sp, -X'
                    if rt == 29 and rs == 29 and (immediate & 0x8000): # $sp is $29, negative immediate
                         main_func_offset = i
                         break

            if main_func_offset is not None:
                main_comment = f"// AI Heuristic: Potential 'main' function prologue found at offset 0x{main_func_offset:08X}\n"
                main_signature = "int main(int argc, char *argv[]) // Simulated Signature\n{\n"
                main_body = " // AI Analysis: Simulated initialization code based on patterns\n"
                main_body += " // TODO: Detailed logic reconstruction would go here\n"
                main_body += " // Example placeholder logic:\n"
                main_body += " initialize_hardware();\n"
                main_body += " game_loop();\n"
                main_body += " return 0;\n"
                main_body += "}\n"
                other_funcs = "// AI Heuristic: Other potential functions identified (stubbed)\n"
                other_funcs += "void initialize_hardware(); // Simulated\n"
                other_funcs += "void game_loop(); // Simulated\n"
            else:
                main_comment = "// AI Heuristic: Could not confidently identify 'main' function.\n"
                main_signature = "void unknown_entry_point() // Simulated\n{\n"
                main_body = " // AI Analysis: Entry point logic is unclear.\n"
                main_body += " // Raw bytes analysis required.\n"
                main_body += "}\n"
                other_funcs = "// No other functions confidently identified.\n"

            full_output = header_comment + main_comment + main_signature + main_body + other_funcs
            full_output += "// - End of Simulated Pseudo C Code -\n"

            with open(output_file, 'w') as f:
                f.write(full_output)

            log(f"Simulated decompilation saved to {output_file}")
            store({
                "action": "pseudo_decompile",
                "rom_path": rom_path,
                "output_path": str(output_file),
                "main_function_offset": f"0x{main_func_offset:08X}" if main_func_offset is not None else "Not Found"
            })
            return f"Simulated decompilation complete. Output saved to: {output_file}\nKey findings:\n- Main function heuristic: {'Found' if main_func_offset is not None else 'Not Found'}"

        except Exception as e:
            log(f"Error during simulated decompilation: {e}")
            return f"Error during simulated decompilation: {e}"

    def find_constants_and_refs(rom_path: str, output_dir: str) -> str:
        """Run simulated AI constant/reference finder."""
        rom_path_obj = pathlib.Path(rom_path)
        if not rom_path_obj.is_file():
            return f"Error: ROM file not found: {rom_path}"

        output_file = pathlib.Path(output_dir) / f"{rom_path_obj.stem}_constants.txt"
        log(f"Starting simulated AI constant/reference finding for {rom_path}...")

        try:
            with open(rom_path, 'rb') as f:
                rom_bytes = f.read()

            output_lines = [f"; Simulated Constants/Refs for {rom_path_obj.name}", f"; File Size: {len(rom_bytes)} bytes", "; --- AI Heuristics Applied ---"]

            # 1. Find potential pointers (addresses within ROM/file)
            pointers_found = []
            rom_size = len(rom_bytes)
            for i in range(0, rom_size - 4, 4):
                word_bytes = rom_bytes[i:i+4]
                # Assume pointers are 32-bit, big-endian, and point within the file
                # This is a very crude heuristic
                try:
                    ptr_val = struct.unpack('>I', word_bytes)[0]
                    # Check if it looks like a plausible pointer
                    # (e.g., aligned, within file bounds, not null)
                    if ptr_val != 0 and ptr_val < rom_size and (ptr_val % 4 == 0):
                        # Further heuristics could check if the pointed-to data looks valid
                        pointers_found.append((i, ptr_val))
                except:
                    pass # Ignore unpacking errors at EOF edge cases

            if pointers_found:
                 output_lines.append("\n; - Potential Pointers Found -")
                 for offset, ptr in pointers_found[:30]: # Limit output
                      output_lines.append(f"{offset:08X}: -> 0x{ptr:08X}")

            # 2. Find potential floating-point constants (IEEE 754)
            floats_found = []
            for i in range(0, rom_size - 4, 4):
                word_bytes = rom_bytes[i:i+4]
                try:
                    # Try interpreting as float (big-endian)
                    f_val = struct.unpack('>f', word_bytes)[0]
                    # Basic check for plausible float (not inf, nan, denormalized usually)
                    if 0.0001 <= abs(f_val) <= 1000000.0:
                        floats_found.append((i, f_val, 'float'))
                    # Try double (8 bytes, so check two words)
                    if i + 8 <= rom_size:
                        double_bytes = rom_bytes[i:i+8]
                        try:
                            d_val = struct.unpack('>d', double_bytes)[0]
                            if 0.0001 <= abs(d_val) <= 1000000.0:
                                floats_found.append((i, d_val, 'double'))
                        except:
                            pass
                except:
                    pass

            if floats_found:
                output_lines.append("\n; - Potential Float/Double Constants Found -")
                for offset, val, ftype in floats_found[:20]: # Limit output
                    output_lines.append(f"{offset:08X}: {val} ({ftype})")

            # 3. Find repeated byte sequences (potential data tables, signatures)
            sequences_found = {}
            seq_len = 16 # Look for 16-byte sequences
            for i in range(0, rom_size - seq_len + 1):
                seq = rom_bytes[i:i+seq_len]
                if seq in sequences_found:
                    sequences_found[seq].append(i)
                else:
                    sequences_found[seq] = [i]

            # Filter for sequences that appear more than once
            repeated_seqs = {seq: offsets for seq, offsets in sequences_found.items() if len(offsets) > 1}
            if repeated_seqs:
                output_lines.append("\n; - Repeated Byte Sequences Found (Potential Data/Signatures) -")
                count = 0
                for seq, offsets in repeated_seqs.items():
                    if count >= 15: # Limit output
                        break
                    output_lines.append(f"Sequence {seq.hex().upper()} found at offsets: {[f'0x{off:08X}' for off in offsets[:5]]}...") # Show first 5
                    count += 1

            output_lines.append("\n; - End of Simulated Constant/Reference Finding -")
            full_output = "\n".join(output_lines) + "\n"

            with open(output_file, 'w') as f:
                f.write(full_output)

            log(f"Simulated constants/references saved to {output_file}")
            store({
                "action": "find_constants",
                "rom_path": rom_path,
                "output_path": str(output_file),
                "pointers_found": len(pointers_found),
                "floats_found": len(floats_found),
                "repeated_seqs": len(repeated_seqs)
            })
            return f"Simulated constant/reference finding complete. Output saved to: {output_file}\nKey findings:\n- Pointers: {len(pointers_found)}\n- Floats/Doubles: {len(floats_found)}\n- Repeated Sequences: {len(repeated_seqs)}"

        except Exception as e:
            log(f"Error during simulated constant/reference finding: {e}")
            return f"Error during simulated constant/reference finding: {e}"

    def run_full_analysis(rom_path: str, output_dir: str) -> str:
        """Run the full suite of simulated analyses."""
        log(f"Running full analysis suite on {rom_path}...")
        results = []
        results.append("--- Full Analysis Suite ---")
        results.append(pseudo_disassemble(rom_path, output_dir))
        results.append(pseudo_decompile(rom_path, output_dir))
        results.append(find_constants_and_refs(rom_path, output_dir))
        results.append("--- End of Full Analysis ---")
        return "\n".join(results)

    def show_history():
        """Display the action history."""
        print("\n--- Recent Actions ---")
        if not ACTION_HISTORY:
            print("No actions recorded yet.")
        else:
            for i, action in enumerate(ACTION_HISTORY):
                print(f"{i+1}. {action}")
        print("---")

    # --- Ghidra Agent Commands ---
    COMMANDS = {
        "load": "Load an N64 ROM file for analysis",
        "disasm": "Run simulated AI disassembly to pseudo-asm on loaded ROM",
        "decompile": "Run simulated AI decompilation to pseudo-C on loaded ROM",
        "constant": "Run simulated AI constant/reference finder on loaded ROM",
        "crypto": "Alias for 'constant' (finds signatures, refs, etc.)",
        "analyze": "Run full suite: disasm, decompile, constant analysis",
        "history": "Show recent actions performed by the agent",
        "exit": "Quit the Ghidra Agent"
    }

    def show_menu():
        print("\n- Ghidra Agent Commands -")
        for cmd, desc in COMMANDS.items():
            print(f" {cmd:<12} - {desc}")
        print("-")

    def get_user_input(prompt: str) -> str:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Ghidra Agent.")
            sys.exit(0)

    # --- Main Ghidra Agent Loop ---
    def agent_loop():
        print("Starting Ghidra Agent...")
        print("This agent uses simulated AI heuristics for ROM analysis as a placeholder.")
        print("For real Ghidra integration, the 'run_headless_analysis' function can be used.")
        show_menu()

        while True:
            command = get_user_input("\nGhidraAgent> ").lower()

            if command == "exit":
                print("Good-bye from Ghidra Agent!")
                break
            elif command == "help":
                show_menu()
            elif command == "load":
                rom_path = get_user_input("Enter path to N64 ROM: ")
                if rom_path:
                    load_rom(rom_path)
            elif command in ["disasm", "decompile", "constant", "crypto"]:
                if not CURRENT_ROM_PATH:
                    print("Error: No ROM loaded. Use 'load' command first.")
                    continue
                if command == "disasm":
                    result = pseudo_disassemble(CURRENT_ROM_PATH, GHIDRA_OUTPUT_DIR)
                elif command == "decompile":
                    result = pseudo_decompile(CURRENT_ROM_PATH, GHIDRA_OUTPUT_DIR)
                elif command in ["constant", "crypto"]:
                    result = find_constants_and_refs(CURRENT_ROM_PATH, GHIDRA_OUTPUT_DIR)
                print(result)
            elif command == "analyze":
                if not CURRENT_ROM_PATH:
                    print("Error: No ROM loaded. Use 'load' command first.")
                    continue
                result = run_full_analysis(CURRENT_ROM_PATH, GHIDRA_OUTPUT_DIR)
                print(result)
            elif command == "history":
                show_history()
            else:
                print(f"Unknown command: {command}. Type 'help' for commands.")

    # Start the agent loop
    agent_loop()
    # Signal that the agent has finished its task
    sys.exit(0) # Exit the entire script when Ghidra agent is done

# --- Cydia Agent (Full Code) ---
# (This is the full cydia_agent function from your uploaded file)
def cydia_agent():
    import os
    import sys
    import subprocess
    import json
    import importlib.util
    import traceback
    from pathlib import Path

    # --- Configuration ---
    CYDIA_PATH = Path("/Users/rosty/Desktop/cydia")
    MODULE_INDEX_FILE = CYDIA_PATH / ".cydia_module_index.json"
    INSTALLED_MODULES_FILE = CYDIA_PATH / ".cydia_installed_modules.json"
    MANUAL_ADD_FOLDER = CYDIA_PATH / "manual_add"

    # Ensure Cydia directory and subdirs exist
    CYDIA_PATH.mkdir(parents=True, exist_ok=True)
    MANUAL_ADD_FOLDER.mkdir(parents=True, exist_ok=True)

    # --- Persistent State ---
    def save_module_index(index):
        """Save the module index to a JSON file."""
        try:
            with open(MODULE_INDEX_FILE, 'w') as f:
                json.dump(index, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save module index: {e}")

    def load_module_index():
        """Load the module index from a JSON file."""
        if MODULE_INDEX_FILE.exists():
            try:
                with open(MODULE_INDEX_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Error] Failed to load module index: {e}")
        return {}

    def save_installed_modules(modules):
        """Save the list of installed modules."""
        try:
            with open(INSTALLED_MODULES_FILE, 'w') as f:
                json.dump(modules, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save installed modules: {e}")

    def load_installed_modules():
        """Load the list of installed modules."""
        if INSTALLED_MODULES_FILE.exists():
            try:
                with open(INSTALLED_MODULES_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Error] Failed to load installed modules: {e}")
        return {}

    # --- Module Management Functions ---
    def scan_and_index_modules():
        """Scan the Cydia directory and create an index of loadable modules."""
        index = {}
        print("[*] Scanning Cydia directory for modules...")
        for item in CYDIA_PATH.iterdir():
            if item.is_file():
                if item.suffix == '.sh':
                    index[item.name] = {'type': 'shell_script', 'path': str(item), 'status': 'indexed'}
                    print(f" [Indexed] Shell script: {item.name}")
                elif item.suffix == '.py':
                    index[item.name] = {'type': 'python_script', 'path': str(item), 'status': 'indexed'}
                    print(f" [Indexed] Python script: {item.name}")
                elif item.suffix == '.deb':
                    index[item.name] = {'type': 'debian_package', 'path': str(item), 'status': 'indexed'}
                    print(f" [Indexed] Debian package: {item.name}")
        save_module_index(index)
        print("[+] Module indexing complete.")
        return index

    def list_modules(indexed_modules, installed_modules):
        """List all indexed and installed modules."""
        print("\n- Indexed Modules -")
        if not indexed_modules:
            print(" No modules found.")
        else:
            for name, info in indexed_modules.items():
                status_marker = "[*]" if name in installed_modules else "[ ]"
                print(f" {status_marker} {name} ({info['type']})")
        print("-")
        print("\n- Installed Modules -")
        if not installed_modules:
            print(" No modules installed.")
        else:
            for name in installed_modules.keys():
                print(f" [*] {name}")
        print("-")

    def install_module(module_name, indexed_modules, installed_modules):
        """Install a module by acknowledging it and potentially making it executable."""
        if module_name not in indexed_modules:
            print(f"[Error] Module '{module_name}' not found in index.")
            return
        if module_name in installed_modules:
            print(f"[!] Module '{module_name}' is already installed.")
            return

        module_info = indexed_modules[module_name]
        module_type = module_info['type']
        module_path = module_info['path']

        try:
            if module_type == 'shell_script':
                os.chmod(module_path, 0o755)
                print(f"[*] Shell script '{module_name}' made executable.")
            elif module_type == 'python_script':
                print(f"[*] Python script '{module_name}' indexed. (Execution logic needed)")
            elif module_type == 'debian_package':
                print(f"[!] Installing .deb packages directly is not standard on macOS.")
                print(f" Consider using Homebrew formulas or converting the package.")
                return

            installed_modules[module_name] = module_info
            save_installed_modules(installed_modules)
            print(f"[+] Module '{module_name}' installed successfully.")
        except Exception as e:
            print(f"[Error] Failed to install module '{module_name}': {e}")

    def uninstall_module(module_name, installed_modules):
        """Uninstall a module (remove from installed list)."""
        if module_name not in installed_modules:
            print(f"[Error] Module '{module_name}' is not installed.")
            return
        try:
            del installed_modules[module_name]
            save_installed_modules(installed_modules)
            print(f"[+] Module '{module_name}' uninstalled.")
        except Exception as e:
            print(f"[Error] Failed to uninstall module '{module_name}': {e}")

    def open_manual_add_folder():
        """Open the manual_add folder in Finder."""
        try:
            subprocess.run(['open', str(MANUAL_ADD_FOLDER)], check=True)
            print("[*] Manual add folder opened. Please rescan modules after adding files.")
        except Exception as e:
            print(f"[Error] Failed to open directory: {e}")

    def execute_shell_module(module_name, indexed_modules):
        """Execute a shell script module."""
        if module_name not in indexed_modules or indexed_modules[module_name]['type'] != 'shell_script':
            print(f"[Error] '{module_name}' is not an indexed shell script.")
            return
        module_path = indexed_modules[module_name]['path']
        try:
            print(f"[*] Executing shell script: {module_path}")
            result = subprocess.run([module_path], capture_output=True, text=True, check=True)
            print(f"[Output]\n{result.stdout}")
            if result.stderr:
                print(f"[Stderr]\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Shell script '{module_name}' failed (exit code {e.returncode}).")
            print(f"[Stdout]\n{e.stdout}")
            print(f"[Stderr]\n{e.stderr}")
        except Exception as e:
            print(f"[Error] Failed to execute shell script '{module_name}': {e}")

    def execute_python_module(module_name, indexed_modules):
        """Load and execute functions from a Python module."""
        if module_name not in indexed_modules or indexed_modules[module_name]['type'] != 'python_script':
            print(f"[Error] '{module_name}' is not an indexed Python script.")
            return
        module_path = indexed_modules[module_name]['path']
        try:
            # Create a unique module name to avoid conflicts
            module_import_name = f"cydia_dynamic_{os.path.splitext(module_name)[0]}"
            spec = importlib.util.spec_from_file_location(module_import_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"[*] Python module '{module_name}' loaded.")

            # List available functions (non-private)
            functions = [name for name, obj in vars(module).items() if callable(obj) and not name.startswith('_')]
            if not functions:
                print(f"[!] No public functions found in '{module_name}'.")
                return
            print(f"Available functions in '{module_name}': {', '.join(functions)}")

            func_choice = input("Enter function name to execute (or press Enter to skip): ").strip()
            if func_choice in functions:
                func_to_call = getattr(module, func_choice)

                args_input = input("Enter arguments as JSON (e.g., {'arg1': 'val1'}), or press Enter for none: ").strip()
                args = ()
                kwargs = {}
                if args_input:
                    try:
                        args_dict = json.loads(args_input)
                        # Separate positional (*args) and keyword (**kwargs)
                        # Convention: use list for *args, dict for **kwargs
                        # If args_dict is a list, treat as *args
                        if isinstance(args_dict, list):
                            args = tuple(args_dict)
                            kwargs = {}
                        # If it's a dict, treat as **kwargs, filtering out potential *args key if needed
                        elif isinstance(args_dict, dict):
                            # Example: Allow {'args': [...], 'kwargs': {...}}
                            if 'args' in args_dict and isinstance(args_dict['args'], list):
                                args = tuple(args_dict['args'])
                            else:
                                args = ()
                            kwargs = {k: v for k, v in args_dict.items() if k != 'args'}
                        else:
                            kwargs = args_dict # Fallback, treat single item as kwarg?
                    except json.JSONDecodeError:
                        print("[Error] Invalid JSON for arguments.")
                        return

                print(f"[*] Calling {func_choice}...")
                try:
                    result = func_to_call(*args, **kwargs)
                    print(f"[Result] {result}")
                except Exception as e:
                    print(f"[Error] Exception occurred in {func_choice}: {e}")
                    traceback.print_exc()
            else:
                if func_choice:
                    print(f"[!] Function '{func_choice}' not found in '{module_name}'.")

        except Exception as e:
            print(f"[Error] Failed to load or execute Python module '{module_name}': {e}")
            traceback.print_exc()

    # --- Cydia Agent Menu ---
    COMMANDS = {
        "1": "Scan and Index Modules",
        "2": "List Modules",
        "3": "Install Module",
        "4": "Uninstall Module",
        "5": "Open 'manual_add' Folder",
        "6": "Execute Shell Module",
        "7": "Execute Python Module",
        "8": "Reload Dynamic Tools (for main agent)",
        "9": "Exit Cydia Agent"
    }

    def show_menu():
        print("\n┌──────────────────────────────┐")
        print("│        Cydia Agent           │")
        print("├──────────────────────────────┤")
        for key, value in COMMANDS.items():
            print(f"│ {key}. {value:<28} │")
        print("└──────────────────────────────┘")

    def get_menu_choice():
        """Get user menu choice."""
        choice = input("Enter your choice: ").strip()
        return choice

    def menu_main():
        """Main menu loop for the Cydia Agent."""
        indexed_modules = load_module_index()
        installed_modules = load_installed_modules()

        while True:
            show_menu()
            choice = get_menu_choice()

            if choice == "1":
                indexed_modules = scan_and_index_modules()
                input("[Press Enter to continue]")
            elif choice == "2":
                list_modules(indexed_modules, installed_modules)
                input("[Press Enter to continue]")
            elif choice == "3":
                list_modules(indexed_modules, installed_modules)
                module_name = input("Enter module name to install: ").strip()
                if module_name:
                    install_module(module_name, indexed_modules, installed_modules)
                input("[Press Enter to continue]")
            elif choice == "4":
                if not installed_modules:
                    print("No modules installed.")
                else:
                    print("- Installed Modules -")
                    for name in installed_modules.keys():
                        print(f" [*] {name}")
                    print("-")
                    module_name = input("Enter module name to uninstall: ").strip()
                    if module_name:
                        uninstall_module(module_name, installed_modules)
                input("[Press Enter to continue]")
            elif choice == "5":
                open_manual_add_folder()
                input("[Press Enter to continue]")
            elif choice == "6":
                list_modules(indexed_modules, installed_modules)
                module_name = input("Enter shell module name to execute: ").strip()
                if module_name:
                    execute_shell_module(module_name, indexed_modules)
                input("[Press Enter to continue]")
            elif choice == "7":
                list_modules(indexed_modules, installed_modules)
                module_name = input("Enter Python module name to execute: ").strip()
                if module_name:
                    execute_python_module(module_name, indexed_modules)
                input("[Press Enter to continue]")
            elif choice == "8": # New option
                print("[*] Reloading dynamic tools from Cydia modules...")
                load_cydia_dynamic_tools() # Reload tools
                print("[+] Dynamic tools reloaded. Restart agents to see new tools.")
                input("[Press Enter to continue]")
            elif choice == "9":
                print("Good-bye from Cydia!")
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Entry Point for Cydia Agent ---
    print(f"[+] Cydia Agent ready. Modules path: {CYDIA_PATH}")
    # Initial scan on startup
    indexed_modules = scan_and_index_modules()
    menu_main()
    cydia_agent()
    sys.exit(0) # Exit the entire script when Cydia agent is done


if __name__ == "__main__":
    select_and_load_model()
    agent_chat()









########################################################################################################
# 5. Cydia Agent
########################################################################################################

def launch_cydia_agent():
    """Wrapper function to launch the Cydia Agent."""
    print("[*] Launching Cydia Agent...")
    cydia_agent() # Launch the agent
    print("[*] Returned from Cydia Agent.")

def cydia_agent() -> None:
    """
    Cydia-like Agent for managing packages/modules within the AI OS.
    Handles .py, .sh, .deb packages, provides a build environment, and integrates with LLMs.
    """
    ORIGINAL_CWD_AGENT = Path.cwd()
    
    # --- Define Dedicated Workspace Directories ---
    WORK = Path.home() / ".cydia_agent"
    WORK.mkdir(parents=True, exist_ok=True)
    
    # Directory for user modules (.py, .sh)
    CYDIA_USER_MODULES_PATH = WORK / "user_modules" 
    CYDIA_USER_MODULES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Directory for .deb packages
    DEBS_DIR = WORK / "debs"
    DEBS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Directory for source/build files (for potential toolchain/package building)
    BUILD_DIR = WORK / "build"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Directory for installed packages (simulated root)
    INSTALL_ROOT = WORK / "install_root" 
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    
    os.chdir(WORK)
    print(f"[*] Cydia Agent workspace: {WORK}")
    print(f"[*] User Modules Path: {CYDIA_USER_MODULES_PATH}")
    print(f"[*] .deb Packages Path: {DEBS_DIR}")
    print(f"[*] Build Workspace: {BUILD_DIR}")
    print(f"[*] Simulated Install Root: {INSTALL_ROOT}")

    # --- Cydia Agent Main Menu System ---
    def menu_main():
        """Main loop for the Cydia Agent menu."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\
┌────────────────────────────────────────────────────────────┐
│ Cydia Agent - Modular Package Management & Toolchain       │
│ Workspace: {str(WORK):<45} │
├────────────────────────────────────────────────────────────┤
│ 1) Manage Modules (.py, .sh)                               │
│    - List, Install, Remove                                 │
│ 2) Manage .deb Packages                                    │
│    - List, Add, Install                                    │
│ 3) Import Files for Build                                  │
│    - Add source files, scripts, resources to build area    │
│ 4) Install Packages (Build-like Operations)                │
│    - Trigger installation of selected packages             │
│ 5) Run                                                     │
│    - Execute installed scripts or tools                    │
│ 6) Build Cydia-like Toolchain (Conceptual)                 │
│ 7) Cydia Chat                                              │
│    - Access specialized chat interface                     │
│ 8) Exit to Main Cyber Agent Menu                           │
└────────────────────────────────────────────────────────────┘
""")
            choice = input("Select [1-8]: ").strip()
            
            # --- Route to Sub-Menus or Functions ---
            if choice == "1":
                menu_manage_modules()
            elif choice == "2":
                menu_manage_debs()
            elif choice == "3":
                import_files_for_build()
            elif choice == "4":
                install_packages()
            elif choice == "5":
                menu_run()
            elif choice == "6":
                build_cydia_toolchain()
            elif choice == "7":
                menu_cydia_chat()
            elif choice == "8":
                print("[*] Returning to main Cyber Agent menu.")
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Sub-Menu: Manage Modules (.py, .sh) ---
    def menu_manage_modules():
        """Sub-menu for managing user modules."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\
┌────────────────────────────────────────────────────────────┐
│ Cydia Agent - Manage Modules (.py, .sh)                    │
├────────────────────────────────────────────────────────────┤
│ 1) List Available Modules (in {CYDIA_USER_MODULES_PATH.name})  │
│ 2) Install Module (.py or .sh)                             │
│ 3) Remove Module                                           │
│ 4) Back to Main Menu                                       │
└────────────────────────────────────────────────────────────┘
""")
            sub_choice = input("Select [1-4]: ").strip()
            if sub_choice == "1":
                list_user_modules()
            elif sub_choice == "2":
                install_user_module()
            elif sub_choice == "3":
                remove_user_module()
            elif sub_choice == "4":
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Sub-Menu: Manage .deb Packages ---
    def menu_manage_debs():
        """Sub-menu for managing .deb packages."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\
┌────────────────────────────────────────────────────────────┐
│ Cydia Agent - Manage .deb Packages                         │
├────────────────────────────────────────────────────────────┤
│ 1) List Downloaded .deb Files (in {DEBS_DIR.name})         │
│ 2) Add .deb Package (Manual File Selection)                │
│ 3) Install .deb Package                                    │
│ 4) Back to Main Menu                                       │
└────────────────────────────────────────────────────────────┘
""")
            sub_choice = input("Select [1-4]: ").strip()
            if sub_choice == "1":
                list_debs()
            elif sub_choice == "2":
                add_deb_manually()
            elif sub_choice == "3":
                install_deb()
            elif sub_choice == "4":
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Sub-Menu: Run Installed Tools/Scripts ---
    def menu_run():
        """Sub-menu for running installed scripts or tools."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\
┌────────────────────────────────────────────────────────────┐
│ Cydia Agent - Run                                          │
│ Executes scripts/tools from {INSTALL_ROOT}                 │
├────────────────────────────────────────────────────────────┤
│ 1) List Runnable Items (Scripts, Binaries)                 │
│ 2) Run Python Script                                       │
│ 3) Run Shell Script                                        │
│ 4) Run Executable/Binary                                   │
│ 5) Back to Main Menu                                       │
└────────────────────────────────────────────────────────────┘
""")
            sub_choice = input("Select [1-5]: ").strip()
            if sub_choice == "1":
                list_runnables()
            elif sub_choice == "2":
                run_python_script()
            elif sub_choice == "3":
                run_shell_script()
            elif sub_choice == "4":
                run_executable()
            elif sub_choice == "5":
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Sub-Menu: Cydia Chat ---
    def menu_cydia_chat():
        """Sub-menu for Cydia Chat functionalities."""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            print(f"""\
┌────────────────────────────────────────────────────────────┐
│ Cydia Agent - Cydia Chat                                   │
├────────────────────────────────────────────────────────────┤
│ 1) Boot Gemini 1.5 Pro Agent (LLM Chat)                    │
│    - Requires .gguf file selection                         │
│ 2) Back to Main Menu                                       │
└────────────────────────────────────────────────────────────┘
""")
            sub_choice = input("Select [1-2]: ").strip()
            if sub_choice == "1":
                boot_gemini_agent()
            elif sub_choice == "2":
                break
            else:
                print("[!] Invalid choice.")
                input("[Press Enter to continue]")

    # --- Core Functionality: Module Management ---
    def list_user_modules():
        """List available user modules (.py, .sh) in the dedicated directory."""
        print(f"[*] Listing modules in '{CYDIA_USER_MODULES_PATH}':")
        found_any = False
        for item in CYDIA_USER_MODULES_PATH.iterdir():
            if item.is_file() and item.suffix in ['.py', '.sh']:
                print(f"  - {item.name}")
                found_any = True
        if not found_any:
            print("  [No .py or .sh modules found.]")
        input("[Press Enter to continue]")

    def install_user_module():
        """Install a .py or .sh module by copying it to the user modules directory."""
        print("[*] Selecting module to install...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            src_file_path_str = filedialog.askopenfilename(
                title="Select Python (.py) or Shell Script (.sh) to Install",
                filetypes=[("Python Files", "*.py"), ("Shell Scripts", "*.sh"), ("All Files", "*.*")]
            )
            root.destroy()
        except Exception:
            src_file_path_str = None

        if not src_file_path_str:
            src_file_path_str = input("Enter full path to the .py or .sh module file: ").strip()

        if not src_file_path_str:
            print("[!] No file selected or entered.")
            input("[Press Enter to continue]")
            return

        src_file_path = Path(src_file_path_str)
        if not src_file_path.is_file() or src_file_path.suffix not in ['.py', '.sh']:
            print(f"[!] Invalid file selected: {src_file_path}. Must be .py or .sh.")
            input("[Press Enter to continue]")
            return

        dest_file_path = CYDIA_USER_MODULES_PATH / src_file_path.name
        try:
            shutil.copy2(src_file_path, dest_file_path)
            print(f"[+] Module '{src_file_path.name}' installed to '{dest_file_path}'.")
            # Make .sh files executable
            if dest_file_path.suffix == '.sh':
                dest_file_path.chmod(dest_file_path.stat().st_mode | stat.S_IEXEC)
                print(f"[+] Made '{dest_file_path.name}' executable.")
        except Exception as e:
            print(f"[!] Failed to install module: {e}")
        input("[Press Enter to continue]")

    def remove_user_module():
        """Remove a user module by deleting it from the user modules directory."""
        py_sh_files = [f for f in CYDIA_USER_MODULES_PATH.iterdir() if f.is_file() and f.suffix in ['.py', '.sh']]
        if not py_sh_files:
            print("[Cydia Agent] No modules found to remove.")
            input("[Press Enter to continue]")
            return

        print("[*] Select module to remove:")
        for i, f in enumerate(py_sh_files):
            print(f"  {i+1}) {f.name}")
        try:
            choice_idx = int(input("Enter number: ")) - 1
            if 0 <= choice_idx < len(py_sh_files):
                file_to_remove = py_sh_files[choice_idx]
                confirmation = input(f"Are you sure you want to remove '{file_to_remove.name}'? (y/N): ").strip().lower()
                if confirmation == 'y':
                    try:
                        file_to_remove.unlink()
                        print(f"[+] Module '{file_to_remove.name}' removed.")
                    except Exception as e:
                        print(f"[!] Failed to remove module: {e}")
                else:
                    print("[*] Removal cancelled.")
            else:
                print("[!] Invalid selection.")
        except ValueError:
            print("[!] Invalid input.")
        input("[Press Enter to continue]")

    # --- Core Functionality: .deb Package Management ---
    def list_debs():
        """List .deb files in the DEBS_DIR."""
        print(f"[*] Listing .deb files in '{DEBS_DIR}':")
        found_any = False
        for item in DEBS_DIR.iterdir():
            if item.is_file() and item.suffix == '.deb':
                print(f"  - {item.name}")
                found_any = True
        if not found_any:
            print("  [No .deb files found.]")
        input("[Press Enter to continue]")

    def add_deb_manually():
        """Manually add a .deb file to the DEBS_DIR."""
        print("[*] Selecting .deb file to add...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            src_file_path_str = filedialog.askopenfilename(
                title="Select .deb Package to Add",
                filetypes=[("Debian Packages", "*.deb"), ("All Files", "*.*")]
            )
            root.destroy()
        except Exception:
            src_file_path_str = None

        if not src_file_path_str:
            src_file_path_str = input("Enter full path to the .deb file: ").strip()

        if not src_file_path_str:
            print("[!] No file selected or entered.")
            input("[Press Enter to continue]")
            return

        src_file_path = Path(src_file_path_str)
        if not src_file_path.is_file() or src_file_path.suffix != '.deb':
            print(f"[!] Invalid .deb file selected: {src_file_path}")
            input("[Press Enter to continue]")
            return

        dest_file_path = DEBS_DIR / src_file_path.name
        try:
            shutil.copy2(src_file_path, dest_file_path)
            print(f"[+] .deb package '{src_file_path.name}' added to '{dest_file_path}'.")
        except Exception as e:
            print(f"[!] Failed to add .deb package: {e}")
        input("[Press Enter to continue]")

    def install_deb():
        """Placeholder for installing a .deb package."""
        deb_files = [f for f in DEBS_DIR.iterdir() if f.is_file() and f.suffix == '.deb']
        if not deb_files:
            print("[Cydia Agent] No .deb files found in the directory.")
            input("[Press Enter to continue]")
            return

        print("[*] Select .deb file to install:")
        for i, f in enumerate(deb_files):
            print(f"  {i+1}) {f.name}")
        try:
            choice_idx = int(input("Enter number: ")) - 1
            if 0 <= choice_idx < len(deb_files):
                deb_to_install = deb_files[choice_idx]
                # --- Conceptual Installation Steps ---
                # 1. Extract the .deb (it's an ar archive)
                #    import arpy # You might need to install this: pip install arpy
                #    with arpy.Archive(deb_to_install) as ar:
                #        ar.extractall(path=WORK / f"tmp_extract_{deb_to_install.stem}")
                # 2. Locate control.tar.gz and data.tar.gz within the extraction
                # 3. Extract control information (metadata)
                # 4. Extract data.tar.gz to the target root filesystem (e.g., INSTALL_ROOT)
                # 5. Handle dependencies (check if required packages are installed)
                # 6. Run preinst/postinst scripts if they exist in the control.tar.gz
                # 7. Update package database (e.g., a simple list file in WORK)
                # 8. Cleanup temporary extraction directory
                # ----------------------------
                print(f"[Cydia Agent] Installing .deb '{deb_to_install.name}' is a complex process.")
                print("[!] This is a placeholder. Actual .deb installation logic would go here.")
                print("[*] Steps would involve:")
                print("    - Extracting the .deb archive (ar format)")
                print("    - Parsing control information")
                print("    - Extracting files to a target directory")
                print("    - Handling dependencies")
                print("    - Running installation scripts")
                # ----------------------------
            else:
                print("[!] Invalid selection.")
        except ValueError:
            print("[!] Invalid input.")
        except Exception as e:
            print(f"[!] Error during .deb installation process: {e}")
            traceback.print_exc()
        input("[Press Enter to continue]")

    # --- Core Functionality: Build Environment ---
    def import_files_for_build():
        """Import files into the build directory for potential compilation/package building."""
        print("[*] Selecting file(s) or directory to import for build...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            # Allow selecting multiple files or a single directory
            src_paths_str = filedialog.askopenfilenames(
                title="Select Files to Import for Build"
            )
            if not src_paths_str: # If no files selected, try directory
                 src_paths_str = [filedialog.askdirectory(title="Select Directory to Import for Build")]
            
            root.destroy()
        except Exception:
            src_paths_str = None

        if not src_paths_str or (len(src_paths_str) == 1 and not src_paths_str[0]):
            src_paths_str = input("Enter full path(s) to file(s) or directory (space-separated): ").strip().split()

        if not src_paths_str or (len(src_paths_str) == 1 and not src_paths_str[0]):
            print("[!] No file(s) or directory selected or entered.")
            input("[Press Enter to continue]")
            return

        imported_count = 0
        for src_path_str in src_paths_str:
            src_path = Path(src_path_str)
            if not src_path.exists():
                print(f"[!] Path does not exist: {src_path}")
                continue
            
            dest_path = BUILD_DIR / src_path.name
            try:
                if src_path.is_file():
                    shutil.copy2(src_path, dest_path)
                    print(f"[+] Imported file '{src_path.name}' to '{dest_path}'.")
                    imported_count += 1
                elif src_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path) # Overwrite existing directory
                    shutil.copytree(src_path, dest_path)
                    print(f"[+] Imported directory '{src_path.name}' to '{dest_path}'.")
                    imported_count += 1
                else:
                    print(f"[!] Skipping special file: {src_path}")
            except Exception as e:
                print(f"[!] Failed to import '{src_path}': {e}")
        
        print(f"[*] Imported {imported_count} item(s) to build directory '{BUILD_DIR}'.")
        input("[Press Enter to continue]")

    # --- Core Functionality: Package Installation (Build-like) ---
    def install_packages():
        """Trigger installation process for selected packages (.py, .sh, .deb)."""
        print("[*] Selecting packages to install (build-like operation)...")
        
        # Gather available packages
        available_py_sh = [f for f in CYDIA_USER_MODULES_PATH.iterdir() if f.is_file() and f.suffix in ['.py', '.sh']]
        available_debs = [f for f in DEBS_DIR.iterdir() if f.is_file() and f.suffix == '.deb']
        
        if not available_py_sh and not available_debs:
            print("[!] No packages (.py, .sh, .deb) available for installation.")
            input("[Press Enter to continue]")
            return

        print("[*] Available packages:")
        all_packages = available_py_sh + available_debs
        for i, pkg in enumerate(all_packages):
            print(f"  {i+1}) {pkg.name}")
        
        try:
            selections_raw = input("Enter numbers to install (space-separated, e.g., '1 3 5'): ").strip()
            if not selections_raw:
                 print("[*] No packages selected.")
                 input("[Press Enter to continue]")
                 return
            selected_indices = [int(s) - 1 for s in selections_raw.split()]
            
            packages_to_install = []
            for idx in selected_indices:
                if 0 <= idx < len(all_packages):
                    packages_to_install.append(all_packages[idx])
                else:
                    print(f"[!] Invalid selection index: {idx + 1}")
            
            if not packages_to_install:
                print("[!] No valid packages selected for installation.")
                input("[Press Enter to continue]")
                return
                
            print(f"[*] Installing {len(packages_to_install)} selected package(s)...")
            for pkg in packages_to_install:
                print(f"  - Installing '{pkg.name}'...")
                if pkg.suffix in ['.py', '.sh']:
                    # Copy to install root or a bin/scripts dir within it
                    install_dest = INSTALL_ROOT / "scripts" / pkg.name
                    install_dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(pkg, install_dest)
                        if pkg.suffix == '.sh':
                            install_dest.chmod(install_dest.stat().st_mode | stat.S_IEXEC)
                        print(f"    [OK] Installed to '{install_dest}'")
                    except Exception as e:
                        print(f"    [ERR] Failed to install {pkg.name}: {e}")
                elif pkg.suffix == '.deb':
                    # Call the .deb install logic (placeholder)
                    # In a real impl, you'd pass the specific deb file
                    print(f"    [INFO] .deb installation requires specific logic. Triggering general .deb installer...")
                    # Temporarily change the deb_files list for install_deb to use this one
                    original_deb_files = [f for f in DEBS_DIR.iterdir() if f.is_file() and f.suffix == '.deb']
                    # This is a hacky way, better to refactor install_deb to accept a path
                    # For now, we just print a message
                    print(f"    [PLACEHOLDER] Would install .deb: {pkg.name}")
                    # install_deb() # This needs refactoring to accept a specific file
                    
            print("[*] Package installation process completed (conceptually).")
            
        except ValueError:
            print("[!] Invalid input. Please enter numbers separated by spaces.")
        except Exception as e:
            print(f"[!] Error during package selection/installation: {e}")
            traceback.print_exc()
            
        input("[Press Enter to continue]")

    # --- Core Functionality: Run Menu Items ---
    def list_runnables():
        """List items that can potentially be run from the install root."""
        scripts_dir = INSTALL_ROOT / "scripts"
        if not scripts_dir.exists():
            print(f"[Run] No 'scripts' directory found in {INSTALL_ROOT}.")
            input("[Press Enter to continue]")
            return
            
        print(f"[*] Listing runnable items in '{scripts_dir}':")
        found_any = False
        for item in scripts_dir.iterdir():
             if item.is_file() and item.suffix in ['.py', '.sh']:
                 print(f"  - {item.name} ({'Executable' if os.access(item, os.X_OK) else 'Script'})")
                 found_any = True
             elif item.is_file(): # Other files
                 print(f"  - {item.name} (File)")
                 found_any = True
        # Could also list binaries if they were installed to a 'bin' dir
        bins_dir = INSTALL_ROOT / "bin"
        if bins_dir.exists():
             for item in bins_dir.iterdir():
                 if item.is_file() and os.access(item, os.X_OK):
                     print(f"  - {item.name} (Binary)")
                     found_any = True
                     
        if not found_any:
            print("  [No runnable items found.]")
        input("[Press Enter to continue]")

    def run_python_script():
        """Run a Python script from the installed scripts directory."""
        scripts_dir = INSTALL_ROOT / "scripts"
        if not scripts_dir.exists():
             print(f"[Run] No 'scripts' directory found.")
             input("[Press Enter to continue]")
             return
             
        py_scripts = [f for f in scripts_dir.iterdir() if f.is_file() and f.suffix == '.py']
        if not py_scripts:
            print("[Run] No Python scripts found in the installed scripts directory.")
            input("[Press Enter to continue]")
            return

        print("[*] Select Python script to run:")
        for i, f in enumerate(py_scripts):
            print(f"  {i+1}) {f.name}")
        try:
            choice_idx = int(input("Enter number: ")) - 1
            if 0 <= choice_idx < len(py_scripts):
                script_to_run = py_scripts[choice_idx]
                print(f"[*] Running Python script: {script_to_run}")
                try:
                    # Use the system's Python interpreter
                    result = subprocess.run([sys.executable, str(script_to_run)], check=True, text=True, capture_output=True)
                    print("--- Script Output ---")
                    print(result.stdout)
                    if result.stderr:
                        print("--- Script Errors ---")
                        print(result.stderr)
                    print("---------------------")
                    print("[+] Script executed successfully.")
                except subprocess.CalledProcessError as e:
                     print(f"[!] Script execution failed with return code {e.returncode}.")
                     print(f"Stdout: {e.stdout}")
                     print(f"Stderr: {e.stderr}")
                except Exception as e:
                     print(f"[!] Error running script: {e}")
            else:
                print("[!] Invalid selection.")
        except ValueError:
            print("[!] Invalid input.")
        input("[Press Enter to continue]")

    def run_shell_script():
        """Run a Shell script from the installed scripts directory."""
        scripts_dir = INSTALL_ROOT / "scripts"
        if not scripts_dir.exists():
             print(f"[Run] No 'scripts' directory found.")
             input("[Press Enter to continue]")
             return
             
        sh_scripts = [f for f in scripts_dir.iterdir() if f.is_file() and f.suffix == '.sh']
        if not sh_scripts:
            print("[Run] No Shell scripts found in the installed scripts directory.")
            input("[Press Enter to continue]")
            return

        print("[*] Select Shell script to run:")
        for i, f in enumerate(sh_scripts):
            print(f"  {i+1}) {f.name}")
        try:
            choice_idx = int(input("Enter number: ")) - 1
            if 0 <= choice_idx < len(sh_scripts):
                script_to_run = sh_scripts[choice_idx]
                print(f"[*] Running Shell script: {script_to_run}")
                # Ensure it's executable (should be from install)
                if not os.access(script_to_run, os.X_OK):
                    print(f"[!] Script '{script_to_run}' is not executable. Attempting to make it executable...")
                    try:
                        script_to_run.chmod(script_to_run.stat().st_mode | stat.S_IEXEC)
                        print("[+] Made script executable.")
                    except Exception as e:
                        print(f"[!] Failed to make executable: {e}")
                        input("[Press Enter to continue]")
                        return
                
                try:
                    # Run the script directly
                    result = subprocess.run([str(script_to_run)], check=True, text=True, capture_output=True, shell=True)
                    print("--- Script Output ---")
                    print(result.stdout)
                    if result.stderr:
                        print("--- Script Errors ---")
                        print(result.stderr)
                    print("---------------------")
                    print("[+] Script executed successfully.")
                except subprocess.CalledProcessError as e:
                     print(f"[!] Script execution failed with return code {e.returncode}.")
                     print(f"Stdout: {e.stdout}")
                     print(f"Stderr: {e.stderr}")
                except Exception as e:
                     print(f"[!] Error running script: {e}")
            else:
                print("[!] Invalid selection.")
        except ValueError:
            print("[!] Invalid input.")
        input("[Press Enter to continue]")

    def run_executable():
        """Run an executable/binary from the install root (e.g., /bin)."""
        bins_dir = INSTALL_ROOT / "bin"
        if not bins_dir.exists():
             print(f"[Run] No 'bin' directory found in {INSTALL_ROOT}.")
             input("[Press Enter to continue]")
             return
             
        executables = [f for f in bins_dir.iterdir() if f.is_file() and os.access(f, os.X_OK)]
        if not executables:
            print("[Run] No executables found in the installed bin directory.")
            input("[Press Enter to continue]")
            return

        print("[*] Select executable to run:")
        for i, f in enumerate(executables):
            print(f"  {i+1}) {f.name}")
        try:
            choice_idx = int(input("Enter number: ")) - 1
            if 0 <= choice_idx < len(executables):
                exe_to_run = executables[choice_idx]
                print(f"[*] Running executable: {exe_to_run}")
                try:
                    # Run the executable
                    result = subprocess.run([str(exe_to_run)], check=True, text=True, capture_output=True)
                    print("--- Executable Output ---")
                    print(result.stdout)
                    if result.stderr:
                        print("--- Executable Errors ---")
                        print(result.stderr)
                    print("-------------------------")
                    print("[+] Executable ran successfully.")
                except subprocess.CalledProcessError as e:
                     print(f"[!] Executable failed with return code {e.returncode}.")
                     print(f"Stdout: {e.stdout}")
                     print(f"Stderr: {e.stderr}")
                except Exception as e:
                     print(f"[!] Error running executable: {e}")
            else:
                print("[!] Invalid selection.")
        except ValueError:
            print("[!] Invalid input.")
        input("[Press Enter to continue]")

    # --- Core Functionality: Build Toolchain (Conceptual) ---
    def build_cydia_toolchain():
        """
        Placeholder/Outline for building a Cydia-like toolchain.
        This is highly complex and project-specific.
        """
        print("[*] Initiating Cydia-like Toolchain Build Process...")
        print("[!] WARNING: This is a conceptual placeholder.")
        print("[!] Building Cydia/Limitless requires significant research.")

        # --- Conceptual Steps (Based on N64 and general build processes) ---

        # 1. Configuration
        CYDIA_REPO_URL = "https://github.com/JohnCoatesOSS/Limitless.git" # Example source
        CYDIA_BUILD_DIR = BUILD_DIR / "cydia_source" # Use BUILD_DIR for source
        CYDIA_INSTALL_PREFIX = INSTALL_ROOT / "cydia_toolchain" # Install within our simulated root
        CYDIA_BUILD_DIR.mkdir(parents=True, exist_ok=True)
        CYDIA_INSTALL_PREFIX.mkdir(parents=True, exist_ok=True)

        # 2. Source Acquisition
        # Requires 'git' to be installed
        if not command_exists("git"):
            print("[!] Error: 'git' is required but not found.")
            input("[Press Enter to continue]")
            return
        # Clone the repository if not already present or force re-clone?
        if not (CYDIA_BUILD_DIR / ".git").exists():
             print(f"[*] Cloning Cydia source from {CYDIA_REPO_URL}...")
             clone_cmd = ["git", "clone", CYDIA_REPO_URL, str(CYDIA_BUILD_DIR)]
             try:
                 subprocess.check_call(clone_cmd)
                 print(f"[*] Cloned Cydia source to {CYDIA_BUILD_DIR}")
             except subprocess.CalledProcessError as e:
                 print(f"[!] Failed to clone repository: {e}")
                 input("[Press Enter to continue]")
                 return
        else:
             print(f"[*] Cydia source directory already exists at {CYDIA_BUILD_DIR}. Skipping clone.")

        # 3. Dependency Analysis
        print("[*] Analyzing build dependencies...")
        # This is the CRITICAL STEP.
        # You need to inspect the Limitless source:
        # - README.md, INSTALL, or BUILD files for instructions
        # - Look for configure.ac, CMakeLists.txt, or similar build scripts
        # - Identify required system libraries (e.g., libapt-pkg-dev, libcurl, gtk)
        # - Determine build tools (make, cmake, autotools)
        print("[!] TODO: Identify and install/build dependencies for Limitless.")
        print("[!] This often requires a macOS environment with Xcode for iOS projects.")
        print("[*] Common dependencies for package managers might include:")
        print("    - libapt-pkg-dev (Advanced Package Tool library)")
        print("    - libcurl4-openssl-dev (for network operations)")
        print("    - libssl-dev (OpenSSL libraries)")
        print("    - Various build tools (build-essential, cmake, autoconf, automake, libtool)")
        print("    - Potentially GUI libraries (libgtk-3-dev, etc.)")
        # Example check for a common dependency (you'd need to check for all required ones)
        # if not command_exists("dpkg"): # Check for a common package manager tool
        #     print("[!] Warning: dpkg not found. You might be missing package management tools.")

        # 4. Build Process
        print("[*] Attempting build process (this is highly speculative)...")
        os.chdir(CYDIA_BUILD_DIR)
        # The process depends entirely on the project's build system.
        # Let's assume it might use CMake as an example (check actual source!)
        build_subdir = CYDIA_BUILD_DIR / "build"
        build_subdir.mkdir(parents=True, exist_ok=True)
        os.chdir(build_subdir)
        cmake_cmd = [
           "cmake", "..",
           f"-DCMAKE_INSTALL_PREFIX={CYDIA_INSTALL_PREFIX}",
           # Add other CMake options you find in the project's build instructions
           # e.g., -DCMAKE_BUILD_TYPE=Release, -DENABLE_GUI=ON, etc.
        ]
        cmake_build_cmd = ["cmake", "--build", ".", "--parallel", "4"]
        cmake_install_cmd = ["cmake", "--install", "."]
        try:
            print(f"[*] Running CMake configure: {' '.join(cmake_cmd)}")
            # subprocess.check_call(cmake_cmd) # Uncomment to try
            print("[*] Running CMake build...")
            # subprocess.check_call(cmake_build_cmd) # Uncomment to try
            print("[*] Running CMake install...")
            # subprocess.check_call(cmake_install_cmd) # Uncomment to try
            print("[+] Cydia toolchain build steps completed (conceptually).")
            print(f"[*] (If successful) Installed to {CYDIA_INSTALL_PREFIX}")
        except subprocess.CalledProcessError as e:
            print(f"[!] CMake build process failed: {e}")
            print("[*] Check the output above and consult the Limitless build documentation.")
        except FileNotFoundError:
             print("[!] Build tool (e.g., cmake) not found. Please install required build tools.")
        finally:
            os.chdir(WORK) # Return to agent workspace

        # 5. Integration
        # print(f"[*] Toolchain installed to {CYDIA_INSTALL_PREFIX}")
        # print("[!] TODO: Integrate built binaries/libraries into AI OS environment.")
        # print("[!] This might involve updating PATH, linking libraries, etc.")

        print("\n[!] Build process is a conceptual example based on common practices.")
        print("[*] Please consult the 'Limitless' repository documentation for actual build steps.")
        input("[Press Enter to continue]")
        os.chdir(WORK) # Ensure we return to the agent's work directory

    # --- Core Functionality: Cydia Chat & LLM Integration ---
    def boot_gemini_agent():
        """Boot the Gemini 1.5 Pro agent by calling its loading function."""
        print("[*] Booting Gemini 1.5 Pro Agent from Cydia Chat...")
        # --- Load the Global Tool Dictionary for Gemini ---
        # This is crucial for the LLM to see the Cydia tools
        # Ensure DYNAMIC_TOOLS is accessible or re-load it here if needed
        # For now, assume tool_dict (which includes CYDIA_DYNAMIC_TOOLS) is the global one
        global tool_dict # Access the main tool dictionary
        # Make sure Cydia dynamic tools are loaded
        load_cydia_dynamic_tools()
        
        # --- Attempt to Load the Gemini Model ---
        try:
            # Import the function from the main script (assuming it's accessible)
            # If it's defined in the same file, you can call it directly.
            # Otherwise, you might need to import or re-define parts of it.
            # Since the function is in the main script, we can call it if it's in scope.
            # However, to avoid re-defining everything, let's assume `load_gemini_model` is accessible.
            # You might need to adjust the import path or structure.
            
            # --- Call the Existing load_gemini_model Function ---
            # This function handles .gguf selection, loading, and starts its own chat loop.
            # We need to pass the current tool context (tool_dict) to it.
            # Modify the `load_gemini_model` signature to accept `DYNAMIC_TOOLS` or use the global.
            
            # Temporarily override DYNAMIC_TOOLS for the LLM to see Cydia tools
            # This assumes load_gemini_model uses a global DYNAMIC_TOOLS or similar
            # If load_gemini_model is in the same scope, it should see the updated tool_dict
            # after load_cydia_dynamic_tools().
            
            # Call the function (assuming it's defined in the same file or imported)
            # We need to ensure it uses the updated tool_dict
            print("[*] Calling load_gemini_model... (Ensure it uses the updated tool_dict)")
            
            # --- Important: This call will start the LLM's own chat loop ---
            # Control will not return here until the LLM chat exits.
            load_gemini_model() # This should be defined in your main script
            
            print("[*] Returned from Gemini 1.5 Pro Agent.")
            
        except NameError:
            print("[!] Error: 'load_gemini_model' function not found in scope.")
            print("[!] Ensure it's defined in the main script and accessible.")
            # Alternative: Re-implement the core loading logic here if needed,
            # but it's better to reuse the existing robust function.
        except Exception as e:
            print(f"[!] Error booting Gemini agent: {e}")
            traceback.print_exc()
        input("[Press Enter to return to the Cydia Chat menu]")

    # --- Entry Point for Cydia Agent ---
    try:
        # Load initial dynamic tools from the global CYDIA_MODULES_PATH
        load_cydia_dynamic_tools()
        menu_main()
    except Exception as e:
        print(f"[Cydia Agent] An unexpected error occurred: {e}")
        traceback.print_exc()
        input("[Press Enter to return to the main menu]")
    finally:
        os.chdir(ORIGINAL_CWD_AGENT)
        print("[*] Cydia Agent finished. Returning to main menu.")


# ... (rest of your existing code, including the main menu where you would add a call to launch_cydia_agent, e.g.,)
# elif choice == "6":
#     launch_cydia_agent()
# elif choice == "7":
#     # ... other options ...


































































































