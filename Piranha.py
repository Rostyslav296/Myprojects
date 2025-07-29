#!/usr/bin/env python3
"""
Modular Agentic Wrapper
- Agents: Gemini, Deepseek, Qwen
- Tools: Centralized in Inventory section
- All agents respond to 'inventory' queries.
- GUI file/folder selection for model loading.
- Each agent has a 'Boot Config' section for safe updates.
"""

# ###############################################################################################
# 1. Agentic Wrapper (Core Imports, UI, Boot Logic)
# ###############################################################################################

import os
import sys
import json
import subprocess
import datetime
import re
import traceback
import time
import inspect
from typing import Dict, Any, Callable

# ANSI Color Codes
GREEN = "\033[92m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Try to import tkinter for GUI dialogs
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Warning: tkinter not available. Falling back to CLI input.")

# Utility
def command_exists(cmd: str) -> bool:
    """Check if a command exists in the system PATH."""
    return subprocess.run(['which', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header(agent_name: str, color: str):
    clear_screen()
    print(f"{color}{BOLD}┌──────────────────────────────────────┐{RESET}")
    print(f"{color}{BOLD}│           {agent_name:^16}           │{RESET}")
    print(f"{color}{BOLD}└──────────────────────────────────────┘{RESET}\n")

def show_thinking(agent_name: str, color: str):
    print(f"{color}{BOLD}{agent_name}{RESET} is thinking", end='', flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end='', flush=True)
    print(f" {RESET}", end='', flush=True)

def get_user_input() -> str:
    print(f"{YELLOW}Enter query{RESET}")
    return input("> ").strip()

# Global variables
BACKEND = None
MODEL = None
MODEL_PATH = None
MODEL_DIR = None

# ###############################################################################################
# 2. Inventory (Shared Tools & Commands)
# ###############################################################################################

import pathlib
import struct

# Paths
SCRIPT_PATH = "/Users/rosty/Desktop/Anacharssis_Z2.py"
DATA_DIR = "/Users/rosty/Desktop/AI/Data"
ACTIVE_MEMORY_DIR = "/Users/rosty/Desktop/AI/Agents/Active_Memory"
COMFY_JSON_PATH = "/Users/rosty/Desktop/AI/Images/comfy_json/comfy.json"

def get_tools_description(tools: Dict[str, Callable]) -> str:
    """Generate a readable description of all tools."""
    descriptions = []
    for name, func in tools.items():
        desc = getattr(func, '__doc__', "No description.")
        try:
            sig = str(inspect.signature(func))
        except:
            sig = "N/A"
        descriptions.append(f"• {name}{sig}: {desc}")
    return "\n".join(descriptions)

# --- Advanced Agentic Tools ---
def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo. Args: query (str)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        return f"Search results for '{query}':\n" + "\n".join([f"  - {r['title']} ({r['href']}): {r['body'][:100]}..." for r in results])
    except ImportError:
        return "Error: 'duckduckgo_search' library not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Error performing web search: {e}"

def read_file(file_path: str) -> str:
    """Read and return the contents of a file. Args: file_path (str)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Content of '{file_path}':\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(file_path: str, content: str) -> str:
    """Write content to a file, overwriting any existing content. Args: file_path (str), content (str)."""
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to '{file_path}'."
    except Exception as e:
        return f"Error writing file: {e}"

def append_file(file_path: str, content: str) -> str:
    """Append content to the end of a file. Args: file_path (str), content (str)."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + content)
        return f"Successfully appended content to '{file_path}'."
    except Exception as e:
        return f"Error appending to file: {e}"

def edit_source_with_nano(changes: Dict[str, str]) -> str:
    """Edit the main script (Anacharssis_Z2.py) with the provided changes. Args: changes (dict of line_number: new_line)."""
    try:
        with open(SCRIPT_PATH, 'r') as f:
            lines = f.readlines()
        for line_key, new_content in changes.items():
            try:
                line_num = int(line_key) - 1
                if 0 <= line_num < len(lines):
                    lines[line_num] = new_content + '\n'
                else:
                    return f"Error: Line {line_num + 1} is out of range (file has {len(lines)} lines)."
            except ValueError:
                return f"Error: Invalid line number key '{line_key}'."
        with open(SCRIPT_PATH, 'w') as f:
            f.writelines(lines)
        # Restart in new terminal
        subprocess.Popen(['open', '-a', 'Terminal', 'python', SCRIPT_PATH])
        return f"Source code updated at {SCRIPT_PATH}. A new terminal has been opened."
    except Exception as e:
        return f"Error editing source: {e}"

def run_shell(command: str) -> str:
    """Execute a shell command and return the output. Args: command (str)."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode == 0:
            return f"Command executed successfully:\n{output}"
        else:
            return f"Command failed with exit code {result.returncode}:\nSTDERR: {error}\nSTDOUT: {output}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression. Args: expression (str)."""
    try:
        # Only allow numbers, operators, and parentheses.
        if re.match(r'^[0-9+\-*/().\s]+$', expression):
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {float(result)}"
        else:
            return "Error: Expression contains invalid characters."
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Populate tool_dict
tool_dict = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "edit_source_with_nano": edit_source_with_nano,
    "run_shell": run_shell,
    "calculate": calculate,
}

# ###############################################################################################
# 3. Gemini Agent (via llama.cpp)
# ###############################################################################################

def load_gemini_model():
    global BACKEND, MODEL, MODEL_PATH
    BACKEND = "llama"
    AGENT_NAME = "Gemini"
    COLOR = GREEN

    # ###############################################################################################
    # Boot Config
    # ###############################################################################################
    print_header(AGENT_NAME, COLOR)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("[ERROR] Install: pip install llama-cpp-python")
        return

    # GUI file selection
    if HAS_TK:
        print("Opening GUI file picker for .gguf model...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        MODEL_PATH = filedialog.askopenfilename(
            title="Select Gemini 1.5 Pro .gguf file",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        root.destroy()
    else:
        MODEL_PATH = input("Enter path to .gguf file: ").strip()

    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        print("[ERROR] No valid file selected.")
        return

    # Suppress ggml logs
    old_stderr = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        MODEL = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=40,
            verbose=False,
            flash_attn=True
        )
    except Exception as e:
        os.dup2(old_stderr, 2)
        os.close(null)
        print(f"[ERROR] Load failed: {e}")
        return
    finally:
        os.dup2(old_stderr, 2)
        os.close(null)
        os.close(old_stderr)

    # ###############################################################################################
    # Agent Loop
    # ###############################################################################################
    def run_agent():
        tools_desc = get_tools_description(tool_dict)
        system_prompt = f"""You are a helpful AI agent.
You have access to these tools:
{tools_desc}

Respond with a JSON array of tool calls if needed: [{{"name": "tool", "arguments": {{}}}}]
If asked about tools, commands, or inventory, describe them clearly.
If no tools are needed, respond directly."""

        history = [{"role": "system", "content": system_prompt}]
        print(f"\n{COLOR}{BOLD}Agent is ready. Type 'exit' to quit.{RESET}\n")

        while True:
            user_input = get_user_input().strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print(f"{COLOR}Goodbye!{RESET}")
                break

            if re.search(r'\b(inventory|tools?|commands?|functions?)\b', user_input, re.I):
                desc = get_tools_description(tool_dict)
                print(f"\n{COLOR}Available tools:\n{desc}{RESET}\n")
                continue

            history.append({"role": "user", "content": user_input})
            prompt = "<|begin_of_text|>"
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

            show_thinking(AGENT_NAME, COLOR)
            response = ""
            try:
                for chunk in MODEL(prompt=prompt, max_tokens=1500, temperature=0.7, stream=True):
                    text = chunk.get('choices', [{}])[0].get('text', '')
                    print(text, end='', flush=True)
                    response += text
                print()
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

            # Clean response
            clean_response = re.sub(r'<\|.*?\|>', '', response).strip()
            history.append({"role": "assistant", "content": response})

            # Parse tool calls
            try:
                match = re.search(r'\[.*\{.*\}.*\]', clean_response, re.DOTALL)
                if match:
                    calls = json.loads(match.group(0))
                    if isinstance(calls, list):
                        results = []
                        for call in calls:
                            name = call.get('name')
                            args = call.get('arguments', {})
                            if name in tool_dict:
                                if 'file_path' in args:
                                    args['file_path'] = os.path.expanduser(args['file_path'].replace('Users/', '/Users/'))
                                result = tool_dict[name](**args)
                                results.append(result)
                                print(f"{GREEN}Tool Result: {result}{RESET}")
                        # Feed results back
                        history.append({"role": "tool", "content": "\n".join(results)})
                        # Final response
                        show_thinking(AGENT_NAME, COLOR)
                        final_prompt = "<|begin_of_text|>"
                        for msg in history:
                            r = msg["role"]
                            c = msg["content"]
                            final_prompt += f"<|start_header_id|>{r}<|end_header_id|>\n{c}<|eot_id|>"
                        final_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
                        final_response = ""
                        for chunk in MODEL(prompt=final_prompt, max_tokens=1500, stream=True):
                            text = chunk.get('choices', [{}])[0].get('text', '')
                            print(text, end='', flush=True)
                            final_response += text
                        print()
                        history.append({"role": "assistant", "content": final_response})
            except Exception as e:
                print(f"\n[TOOL ERROR] {e}")

    # ✅ Call run_agent() after everything is defined
    run_agent()






# ###############################################################################################
# 4. Deepseek Agent (via ollama)
# ###############################################################################################

def load_deepseek_model():
    global BACKEND, MODEL
    BACKEND = "ollama"
    AGENT_NAME = "Deepseek"
    COLOR = BLUE

    # ###############################################################################################
    # Boot Config
    # ###############################################################################################
    print_header(AGENT_NAME, COLOR)

    if not command_exists('ollama'):
        print("Ollama not found. Install: https://ollama.com/download")
        sys.exit(1)

    MODEL = "deepseek-r1:7b"
    try:
        subprocess.check_call(['ollama', 'show', MODEL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Pulling {MODEL}...")
        subprocess.check_call(['ollama', 'pull', MODEL])

    # ###############################################################################################
    # Agent Loop
    # ###############################################################################################
    def run_agent():
        tools_desc = get_tools_description(tool_dict)
        system_prompt = f"""You are a helpful AI agent.
You have access to these tools:
{tools_desc}

Respond with a JSON array of tool calls if needed: [{{"name": "tool", "arguments": {{}}}}]
If asked about tools or inventory, describe them clearly."""

        history = [{"role": "system", "content": system_prompt}]
        print(f"\n{COLOR}{BOLD}Agent is ready. Type 'exit' to quit.{RESET}\n")

        while True:
            user_input = get_user_input().strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print(f"{COLOR}Goodbye!{RESET}")
                break

            if re.search(r'\b(inventory|tools?|commands?|functions?)\b', user_input, re.I):
                desc = get_tools_description(tool_dict)
                print(f"\n{COLOR}Available tools:\n{desc}{RESET}\n")
                continue

            history.append({"role": "user", "content": user_input})
            prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history] + ["Assistant: "])

            show_thinking(AGENT_NAME, COLOR)
            try:
                result = subprocess.run(
                    ['ollama', 'generate', MODEL],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=300,
                    check=True
                )
                response = result.stdout.strip()
                print(response)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

            history.append({"role": "assistant", "content": response})

    run_agent()

# ###############################################################################################
# 5. Qwen Agent (via transformers)
# ###############################################################################################

def load_qwen_model():
    global BACKEND, MODEL, MODEL_DIR
    BACKEND = "transformers"
    AGENT_NAME = "Qwen"
    COLOR = PURPLE

    # ###############################################################################################
    # Boot Config
    # ###############################################################################################
    print_header(AGENT_NAME, COLOR)

    # GUI folder selection
    if HAS_TK:
        print("Opening GUI folder picker for Qwen model...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        MODEL_DIR = filedialog.askdirectory(
            title="Select Qwen Model Directory"
        )
        root.destroy()
    else:
        MODEL_DIR = input("Enter Qwen model path (or 'Qwen/Qwen3-8B'): ").strip()

    if not MODEL_DIR:
        print("[ERROR] No directory selected.")
        return
    if not os.path.isdir(MODEL_DIR):
        print(f"[ERROR] Directory not found: {MODEL_DIR}")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "transformers"])
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        MODEL = {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        return

    # ###############################################################################################
    # Agent Loop
    # ###############################################################################################
    def run_agent():
        tools_desc = get_tools_description(tool_dict)
        system_prompt = f"""You are a helpful AI agent.
You have access to these tools:
{tools_desc}

Respond with a JSON array of tool calls if needed.
If asked about your inventory, describe it clearly."""

        history = [{"role": "system", "content": system_prompt}]
        print(f"\n{COLOR}{BOLD}Agent is ready. Type 'exit' to quit.{RESET}\n")

        while True:
            user_input = get_user_input().strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print(f"{COLOR}Goodbye!{RESET}")
                break

            if re.search(r'\b(inventory|tools?|commands?|functions?)\b', user_input, re.I):
                desc = get_tools_description(tool_dict)
                print(f"\n{COLOR}Available tools:\n{desc}{RESET}\n")
                continue

            history.append({"role": "user", "content": user_input})
            try:
                inputs = MODEL["tokenizer"](
                    tokenizer.apply_chat_template(history, tokenize=False),
                    return_tensors="pt"
                ).to(MODEL["model"].device)

                show_thinking(AGENT_NAME, COLOR)
                outputs = MODEL["model"].generate(**inputs, max_new_tokens=1500)
                response = MODEL["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                print(response)
                history.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"\n[ERROR] {e}")
                continue

    run_agent()

# ###############################################################################################
# 6. Boot Menu
# ###############################################################################################

if __name__ == "__main__":
    print(f"{GREEN}Gemini{RESET}  |  {BLUE}Deepseek{RESET}  |  {PURPLE}Qwen{RESET}")
    print("Select Agent to Boot:")
    print(f"1. {GREEN}Gemini 1.5 Pro{RESET} (via llama.cpp)")
    print(f"2. {BLUE}Deepseek R1 7B{RESET} (via ollama)")
    print(f"3. {PURPLE}Qwen{RESET} (via transformers)")
    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        load_gemini_model()
    elif choice == "2":
        load_deepseek_model()
    elif choice == "3":
        load_qwen_model()
    else:
        print("Invalid choice.")
        sys.exit(1)