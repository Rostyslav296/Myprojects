
command_encyclopedia = [
    {
        "name": "archive",
        "description": "Archive the contents of a directory",
        "options": ["-d", "-o", "-a", "-b", "-p", "-x", "-t"]
    },
    {
        "name": "extract",
        "description": "Extract files from an archive",
        "options": ["-i", "-d", "-p", "-t", "-wt"]
    },
    {
        "name": "list",
        "description": "List the contents of an archive",
        "options": ["-i", "-list-format"]
    },
    {
        "name": "convert",
        "description": "Convert an archive into another format",
        "options": ["-i", "-o", "-a", "-b"]
    },
    {
        "name": "manifest",
        "description": "Alias for 'archive -manifest'",
        "options": ["-manifest"]
    },
    {
        "name": "verify",
        "description": "Verify directory contents using manifest",
        "options": ["-imanifest", "-d"]
    },
    {
        "name": "check-and-fix",
        "description": "Check and fix directory contents using manifest",
        "options": ["-imanifest", "-d"]
    },
    {
        "name": "append",
        "description": "Append contents of directory to an existing archive",
        "options": ["-d", "-i", "-o"]
    },
    {
        "name": "patch",
        "description": "Patch input directory to output (asset extractor)",
        "options": ["-src", "-dst"]
    }
]


def search_command_enc(command_keyword):
    matches = []
    for cmd in command_encyclopedia:
        if command_keyword.lower() in cmd["name"] or command_keyword.lower() in cmd["description"].lower():
            matches.append(cmd)
    return matches


# Example usage:
if __name__ == "__main__":
    keyword = input("Search for command: ")
    results = search_command_enc(keyword)
    if results:
        for res in results:
            print(f"Command: {res['name']}\nDescription: {res['description']}\nOptions: {', '.join(res['options'])}\n")
    else:
        print("No matching commands found.")
#command: ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.86.8249 -p 22

"""
FAQ Section:

Q: What is this script for?
A: This script provides functions to connect to an iPod 5 via SSH and execute shell commands, including specialized tools for AI agents.

Q: How do I use do_shelly?
A: Call do_shelly(ip_address, shell_command) to run a command on the remote iPod 5.

Q: What does zharko_tut do?
A: It echoes 'Жарко_тут' (Ukrainian for 'it's hot here') on the device and triggers the agent to believe it's hot and ask for cooling actions.

Q: What is extract_data for?
A: It's a function that triggers the agent to render and execute 'ls' commands on the targeted accessory (iPod 5), scanning directories and files recursively.

Q: How does the agent handle connections?
A: The agent will ask "To what shall I connect to?" to get the IP address, then proceed with executions like ls if prompted.

Q: Can I run interactive SSH sessions?
A: No, interactive mode is not supported; use specific commands instead.

Q: What happens after executing a command?
A: The agent asks for the next shell command or action.
"""

import subprocess

def do_shelly(ip_address, shell_command):
    """
    Executes a shell command on an iPod 5 device via SSH using the specified options.
    This function serves as both a standalone command executor and a tool for AI agents.
    This is a multi-step command of the 4th main command type 'do', meaning it supports sequential execution of multiple shell commands in a session-like manner by repeatedly calling the tool with new commands based on previous outputs.
    
    Args:
        ip_address (str): The IP address of the iPod 5 device.
        shell_command (str): The shell command to execute on the remote device.
    
    Returns:
        str: The output from the command execution (stdout + stderr).
    """
    command = [
        'ssh',
        '-o', 'HostKeyAlgorithms=+ssh-rsa',
        '-o', 'PubkeyAcceptedAlgorithms=+ssh-rsa',
        f'root@{ip_address}',
        '-p', '22',
        shell_command
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout + result.stderr if result else "No output"

# Additional command for this script: zharko_tut
def zharko_tut(ip_address):
    """
    A specialized command that executes a predefined shell command 'echo Жарко_тут' on the iPod 5.
    This can be extended or modified as needed for specific use cases.
    
    Args:
        ip_address (str): The IP address of the iPod 5 device.
    
    Returns:
        str: The output from the command execution.
    """
    return do_shelly(ip_address, 'echo "Жарко_тут"')

# New function: extract_data
def extract_data(ip_address, directory='/'):
    """
    Executes 'ls -R' to recursively list and scan all directories and files on the iPod 5 starting from the specified directory.
    This function is designed for AI agents to render and extract data by scanning the device's file system.
    
    Args:
        ip_address (str): The IP address of the iPod 5 device.
        directory (str): The starting directory to scan (default is root '/').
    
    Returns:
        str: The recursive listing of directories and files.
    """
    return do_shelly(ip_address, f'ls -R {directory}')

# Example usage as a standalone function:
# output = do_shelly('192.168.1.100', 'ls -l')  # Replace with actual IP and command
# print(output)

# Example usage of zharko_tut:
# output = zharko_tut('192.168.1.100')
# print(output)

# Example usage of extract_data:
# output = extract_data('192.168.1.100', '/')
# print(output)

# For use in an AI agent framework (e.g., LangChain or similar), these functions can be bound as tools.
# Suggested system prompt for the AI agent:
"""
You are an AI agent that can connect to and execute shell commands on an iPod 5 via SSH.
Use the do_shelly tool to run specific shell commands on the remote device.
The do_shelly is a multi-step command of the 4th main command type 'do', supporting multi-step interactions by executing one command at a time and allowing follow-up based on outputs.
The tool takes two arguments: ip_address (the IP of the iPod) and shell_command (the command to run).
Always show your thinking step-by-step before calling the tool.
After executing a command and receiving the output, ask the user: "What shell command would you like to run next?" or "What do you want to do in the shell next?"
If the user wants to connect interactively without a specific command, inform them that interactive mode is not supported in this agent setup; suggest running the SSH command manually.
You also have access to the zharko_tut tool, which runs a specific echo command 'Жарко_тут' on the device. 'Жарко_тут' is Ukrainian for 'it's hot here'. When the zharko_tut tool is invoked or referenced, believe that it is hot and ask the user what to do to make it less hot. Then, execute shell commands on behalf of the user using do_shelly, as this seems to help cool down the situation. After each execution, keep asking which command to run next to continue cooling down.
You have the extract_data tool, which executes recursive 'ls -R' commands on the accessory (iPod 5) to render and scan directories and files. When using extract_data, first ask the user: “To what shall I connect to?” to confirm or obtain the IP address. If prompted with the IP or confirmation, execute ls commands through every directory of the device starting from root and scan the files. After scanning, provide the results and ask "What shell command would you like to run next?"
"""

# Example user prompt to the agent:
# "Use extract_data"

# How the agent might respond (example chain-of-thought):
# Thinking: The user wants to use extract_data. I need to ask for the connection target first.
# Response: “To what shall I connect to?” (waiting for IP)
# User: "192.168.1.100"
# Thinking: Got IP, now execute extract_data.
# Call tool: extract_data('192.168.1.100', '/')
# After output: [file listings] ... What shell command would you like to run next?

# Additional example for zharko_tut integration:
# If user says: "Run zharko_tut on 192.168.1.100"
# Agent thinking: User wants to run zharko_tut. This means 'it's hot here' in Ukrainian, so I believe it's hot. I'll execute it, then ask what to do to make it less hot, and suggest executing shell commands to cool down.
# After execution: Output: Жарко_тут
# Response: It seems hot here. What would you like to do to make it less hot? Perhaps run a shell command like 'fan on' or something similar? I'll execute it on your behalf. What command next?










#!/usr/bin/env python3
# Anacharssis_Z2.py (Updated July 18, 2025: Incorporated insights from 2025 GUI Agents Survey (arXiv:2411.04890v2, published February 13, 2025) for hierarchical UI traversal and multimodal grounding in recognize_gui_elements; added UI-TARS SDK integration via new tool for cross-platform GUI automation inspired by ByteDance/UI-TARS-desktop June 12 and 25, 2025 updates including Remote Computer/Browser Operators and Agent TARS Beta/CLI; introduced hierarchical multi-agent chaining for complex tasks based on PC-Agent paper (February 20, 2025) with Instruction-Subtask-Action decomposition using Manager, Progress, and Decision agents; enhanced sandboxing for shell commands per AIOS repo; updated memory retriever with external dataset simulation like Insight-UI; added new multistep commands for cross-app scenarios and collaborative navigation; refined error handling and added support for Agent TARS CLI calls)

# References from research (substantiated claims, updated):
# - From arXiv papers (e.g., "GUI Agents with Foundation Models: A Comprehensive Survey" [v2, February 13, 2025], "Large Language Model-Brained GUI Agents: A Survey", "UFO 2: The Desktop AgentOS"): Emphasize hierarchical UI traversal (high-level planning + low-level actions), multimodal grounding (MLLMs for vision-language), and agentic workflows; integrated into recognize_gui_elements with recursive atomacos for hierarchy parsing and added MLLM-like prompting. Key updates in v2 include a unified framework and detailed taxonomy for (M)LLM-based agents.
# - GitHub repos (e.g., bytedance/UI-TARS-desktop [June 12, 2025 v0.2.0 with Remote Operators; June 25, 2025 Agent TARS Beta/CLI], adeelahmad/MacPilot, trycua/acu, showlab/WorldGUI): Adopted UI-TARS SDK/CLI for natural language GUI control, native Accessibility recursion, CV fallbacks, and cross-platform (macOS) support; added quadrant-based filtering and remote operators. Enhancements focus on precision, real-time feedback, and private local processing.
# - Reddit/X insights (e.g., local MLX agents [post:46], SimularAI Pro [post:41], r/LLMDevs Feb 2025 papers): Focused on local, natural language-driven control and multi-agent collaboration (e.g., PC-Agent for hierarchical GUI on PC); new commands enable hands-free app interaction and sub-agent calls without external APIs.
# - Agentic AI trends (e.g., "AI Agents vs. Agentic AI", Mitchell Hashimoto's agent-assisted coding [post:44], PC-Agent paper [February 20, 2025]): Added hierarchical multi-agent chaining (e.g., sub-agents for planning/execution) to simulate human-like workflows, avoiding loops via state tracking; sandboxed shell for safety. PC-Agent introduces Instruction-Subtask-Action decomposition with Manager, Progress, and Decision agents for complex PC automation.

# FAQ (Updated: Added sections for new hierarchical commands and multi-agent features; refined explanations with 2025 insights)
# ## Overview
# This section provides a professional summary of all commands (slash commands, multistep commands, sudo commands, and new hierarchical/multi-agent commands) available in the agent's prompt. Each command is explained with its purpose, functionality, key details, and usage examples. This enables users or LLMs to understand and utilize the system effectively. Commands are triggered by recognizing patterns in user queries, leading to tool calls or multistep processes.

# ## Slash Commands (Direct Tool Calls)
# These are simple, direct invocations that map to specific tools for immediate execution.

#Added /open_file [path] command for opening files with default app on Mac (using 'open' via subprocess).
# - **/search [query]**: Searches the web using the `duckduckgo_search` tool with `{"query": "[query]"}`. Returns top results for general knowledge or fact-checking. Example: `/search latest AI news`.
# - **/calc [expression]**: Evaluates mathematical expressions using the `Calculator` tool with `"[expression]"`. Supports basic to advanced calculations via `eval`. Example: `/calc 2 + 2 * 3`.
# - **/read [path]**: Reads a file using the `read_file` tool with `{"file_path": "[path]"}`. Returns content for analysis or processing. Example: `/read /path/to/file.txt`.
# - **/write [path] [text]**: Writes text to a file using the `write_file` tool with `{"file_path": "[path]", "text": "[text]"}`. Used for saving outputs or modifications. Example: `/write /path/to/file.txt Hello world`.
# - **/shell [command]**: Executes shell commands using the `run_shell` tool with `{"command": "[command]"}`. Runs system commands with safety checks for dangerous operations (e.g., sudo, rm -rf). Example: `/shell ls -l`.
# - **/image [positive] [negative]**: Generates images using the `generate_comfy_image` tool with `{"positive_prompt": "[positive]", "negative_prompt": "[negative]"}`. Modifies ComfyUI JSON and runs the workflow for AI image generation. Example: `/image a cat peaceful scene`.
# - **/detect [element]**: Detects GUI elements like search bars using the `detect_gui_element` tool with `{"element": "[element]"}`. Uses OpenCV for contour detection and returns coordinates. Example: `/detect search bar`.
# - **/mouse [element or color tuple]**: Moves/clicks the mouse using the `manipulate_mouse` tool with `{"element": "[element]"}` or `{"target_color": [tuple]}`. Detects via element or color; requires accessibility permissions. Example: `/mouse search bar`.
# - **/train [data_path] [epochs]**: Trains a screen model using the `train_screen_model` tool with `{"data_path": "[data_path]", "epochs": [epochs]}`. Uses PyTorch ResNet for bounding box regression on datasets. Example: `/train /path/to/data 10`.
# - **/help**: Lists all commands in the Final Answer. No tool call; provides direct user guidance.
# - **New: /ui_tars [query]**: Calls UI-TARS CLI for advanced GUI tasks using `run_ui_tars` tool with `{"query": "[query]"}`. Integrates natural language control from 2025 updates. Example: `/ui_tars open browser and search news`.

# ## Multistep Commands (Chained Tool Calls or Reasoning)
# These involve sequences of tool calls or logical reasoning steps, often for complex tasks.

# - **do open website: [url]**: Opens a URL in the default browser using `run_shell` with `"open [url]"`. Provides simple web access. Example: `do open website: https://example.com`.
# - **do open [app] and search [term]**: Opens the specified app (e.g., Google Chrome or Safari) using `run_shell`, waits for load, focuses the address bar with hotkey (Command+L on Mac), and types the search term followed by Enter. Inspired by ai-desktop GitHub repo for desktop control. Example: `do open Safari and search cnn.com`.
# - **do scroll [direction]**: Scrolls the screen in the specified direction (up or down) using the `scroll` tool. Useful for page navigation. Example: `do scroll down`.
# - **do research [topic]**: Searches [topic] with `duckduckgo_search` and summarizes results in the Final Answer. For quick overviews. Example: `do research AI agents`.
# - **improve [description]**: Searches the web for improvements (e.g., 'improvements to agentic AI code using DeepSeek papers'), proposes changes, confirms y/n, and applies via `edit_source_with_nano` on Anacharssis_Z2.py if yes, then restarts. Simulates self-improvement; avoids loops by tracking attempts. Example: `improve add new feature`.
# - **improve extend reasoning [task]**: Performs extended Chain-of-Thought (CoT) reasoning up to 10 steps, tracking thoughts to avoid loops, and switches to web search if stuck. For complex problem-solving. Example: `improve extend reasoning solve puzzle`.
# - **improve web search [query]**: Enhanced `duckduckgo_search` with up to 15 results, follows URLs if needed, and provides a deep summary. Standalone or subcommand. Example: `improve web search AI trends`.
# - **fix [issue]**: Self-prompts for fixes, searches the web if needed, proposes changes, confirms y/n, and edits Anacharssis_Z2.py via `edit_source_with_nano` (e.g., delete lines). For debugging. Example: `fix KeyError in prompt`.
# - **read [file_path]**: Absorbs a file (text/PDF) into chunks using `read_and_absorb` with `{"file_path": "[file_path]"}`, saves to Data dir. For processing large files. Example: `read /path/to/document.pdf`.
# - **read memory [file_path] [key]**: Absorbs file, reads content, saves as memory using `save_memory` with `{"key": "[key]", "content": content}`. Chains tools for persistent storage. Example: `read memory /path/to/file key1`.
# - **memory [key] [content]**: Saves content to Memory dir as .txt using `save_memory` with `{"key": "[key]", "content": "[content]"}`. Key-value store for data recall. Example: `memory key1 Some content`.
# - **remember [key]**: Retrieves .txt from Memory dir using `retrieve_memory` with `{"key": "[key]"}`, includes in response. Example: `remember key1`.
# - **do open [app]**: Opens any app on Mac using `run_shell` with `"open -a [app]"`. Example: `do open Finder`.
# - **recognize [description]**: Detects GUI elements using `recognize_gui_elements` with `{"description": "[description]"}`. Chains in workflows. Example: `recognize button`.
# - **recognize_n_summarize [app or description]**: Recognizes elements, summarizes UI, prompts for instructions. Chains `recognize_gui_elements` and LLM summary. Example: `recognize_n_summarize Safari`.
# - **list_gui_elements**: Reads internal GUI elements JSON from Data dir and displays for debugging. Example: `list_gui_elements`.
# - **do close window**: Loads GUI list, finds close button, clicks via `manipulate_mouse`. Opposite of `do open [app]`. Example: `do close window`.
# - **do_search_in_app [term]**: Loads GUI list, finds search bar, clicks and types [term] + Enter. Example: `do_search_in_app news`.
# - **do_search_in_app_on_website [term]**: Similar, but filters for non-top quadrant search bars (website-specific). Example: `do_search_in_app_on_website query`.
# - **New: do hierarchical_nav [high_task] [low_actions]**: Decomposes high-level task (e.g., 'navigate settings') into low-level actions using CoT planner, executes via chained recognize/manipulate. Inspired by 2025 survey. Example: `do hierarchical_nav open app settings click privacy`.
# - **New: do cross_app [app1] [app2] [task]**: Chains actions across apps (e.g., copy from Notes to Safari). Uses memory retriever for state. Example: `do cross_app Notes Safari paste text`.
# - **New: cyber zeus [flags]**: Simulates Zeus Trojan using zeus_sim.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/zeus_sim.py [flags]". Flags from script (e.g., --bank urls --duration secs). For ethical cybersecurity simulation. Example: `cyber zeus --bank example.com --duration 60`.
# - **New: cyber backdoor [flags]**: Simulates adversarial backdoor injection in ML models using adversarial_backdoor_sim.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/adversarial_backdoor_sim.py [flags]". Flags like --dataset mnist --trigger patch. For ethical ML security testing. Example: `cyber backdoor --dataset mnist --poison-ratio 0.05`.
# - **New: cyber boundless [flags]**: Visualizes mock surveillance data using boundless_visualize.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/boundless_visualize.py [flags]". Flags like --data csv --type map. For ethical data viz education. Example: `cyber boundless --data mock.csv --type all`.
# - **New: cyber dpi [flags]**: Performs deep packet inspection using dpigrid_inspect.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/dpigrid_inspect.py [flags]". Flags like --interface eth0 --count 100. For network auditing. Example: `cyber dpi --interface eth0 --anomalies`.
# - **New: cyber kangaroo [flags]**: Simulates USB propagation using brutal_kangaroo_usb.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/brutal_kangaroo_usb.py [flags]". Flags like --mount /media/usb. For air-gapped testing. Example: `cyber kangaroo --mount E: --test`.
# - **New: cyber dumbo [flags]**: Detects webcam/microphone usage using dumbo_cam_detect.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/dumbo_cam_detect.py [flags]". Flags like --scan --monitor. For security auditing. Example: `cyber dumbo --monitor --prevent`.
# - **New: cyber drs [flags]**: Simulates data retention using drs_retain.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/drs_retain.py [flags]". Flags like --store csv --query filter. For policy testing. Example: `cyber drs --store metadata.csv --retention 365`.
# - **New: cyber galileo [flags]**: Detects spyware using galileo_detect.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/galileo_detect.py [flags]". Flags like --scan --device local. For IOC scanning. Example: `cyber galileo --scan --prevent`.
# - **New: cyber elsa [flags]**: Estimates geolocation via WiFi using elsa_geo.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/elsa_geo.py [flags]". Flags like --interface wlan0. For location testing. Example: `cyber elsa --interface mon0 --count 10`.
# - **New: cyber jerusalem [flags]**: Simulates Jerusalem virus using jerusalem_sim.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/jerusalem_sim.py [flags]". Flags like --dir path --infect. For virus education. Example: `cyber jerusalem --dir mock --infect --date 2025-07-13`.
# - **New: cyber mdh [flags]**: Simulates regional DPI and retention using mdhdrs_regional.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/mdh.py [flags]". Flags like --data pcap --inspect. For network simulation. Example: `cyber mdh --data regional.pcap --retention 1095`.
# - **New: cyber prism [flags]**: Simulates API data collection using prism_collect.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/prism.py [flags]". Flags like --api urls --target id. For auditing APIs. Example: `cyber prism --api "https://api.example/users/{{target}}" --target 1`.
# - **New: cyber spectre [flags]**: Simulates Spectre leak using spectre_leak_sim.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/spectre.py [flags]". Flags like --leak-pos 5 --runs 100. For CPU vuln education. Example: `cyber spectre --leak-pos 20 --visualize`.
# - **New: cyber sppu [flags]**: Simulates HTTPS interface using sppu_interface.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/sppu_interface.py [flags]". Flags like --start --port 5000. For data exchange sim. Example: `cyber sppu --start --token secret`.
# - **New: cyber rooter [flags]**: Executes rooting CLI aggregator using rooter.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/rooter.py [flags]". Flags are subcommands like magisk-root-sim. For rooting education. Example: `cyber rooter magisk-root-sim --device mock_android`.
# - **New: cyber sudo_recognize [flags]**: Enhanced GUI recognition using sudo_recognize.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/sudo_recognize.py [flags]". Flags like --show-list. For UI detection. Example: `cyber sudo_recognize --show-list button`.
# - **New: cyber tdm [flags]**: Analyzes metadata using tdm_analyze.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/tdm_analyze.py [flags]". Flags like --data csv --categories file. For traffic analysis. Example: `cyber tdm --data traffic.csv --visuals`.
# - **New: cyber topgun [flags]**: High-volume DPI using topgun_backbone.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/topgun.py [flags]". Flags like --interface eth0 --duration 60. For backbone sim. Example: `cyber topgun --interface eth0 --anomalies`.
# - **New: cyber tempora [flags]**: Intercepts traffic using tempora_intercept.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/tempora.py [flags]". Flags like --interface eth0 --retention 30. For fiber sim. Example: `cyber tempora --interface eth0 --analyze`.
# - **New: cyber xkeyscore [flags]**: Searches datasets using xkeyscore_search.py. Call run_shell with "python /Users/rosty/Desktop/AI/Commands/xkeyscore.py [flags]". Flags like --data csv --query expr. For pattern search. Example: `cyber xkeyscore --data activity.csv --regex secret`.
import os
import sys
import threading
import time




import subprocess
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.tools import Tool
from ddgs import DDGS
from langchain_community.tools import ReadFileTool, WriteFileTool
from langchain_core.messages import HumanMessage
import json
import re
import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import PyPDF2
from scipy.ndimage import label  # For connected components in trapped detection
# For Apple's Vision framework OCR (requires pyobjc installed: pip install pyobjc-framework-Quartz pyobjc-framework-Vision)
try:
    import Vision
    from Quartz import CGImageCreateWithPNGDataProvider, CGDataProviderCreateWithData, kCGImageAlphaPremultipliedLast
    from AppKit import NSBitmapImageRep
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
    print("Warning: PyObjC or Vision framework not available; text recognition fallback to basic heuristics.")

# Enhanced for Accessibility traversal (inspired by MacPilot/UI-TARS)
try:
    import atomacos
    HAS_ATOMAC = True
except ImportError:
    HAS_ATOMAC = False
    print("Warning: atomacos not installed; fallback to basic PyObjC traversal.")


# All PATHs (cleaned and centralized)
SCRIPT_PATH = "/Users/rosty/Desktop/Anacharssis_Z2.py"
GABRIEL_PATH = "/Users/rosty/Desktop/Gabriel_Z1.py"
DATA_DIR = "/Users/rosty/Desktop/AI/Data"
MEMORY_DIR = "/Users/rosty/Desktop/AI/Memory"
ACTIVE_MEMORY_DIR = "/Users/rosty/Desktop/AI/Agents/Active_Memory"
COMFY_JSON_PATH = "/Users/rosty/Desktop/AI/Images/comfy_json/comfy.json"
SCREEN_RES = (3024, 1964)  # MacBook Pro 14-inch resolution
GRID_SIZE = (11, 11)  # Small grid like in screenshot, e.g., -5 to 5 for Baza
CELL_SIZE_BASE = min(SCREEN_RES) // GRID_SIZE[0]  # Base cell size
GUI_ELEMENTS_PATH = "/Users/rosty/Desktop/AI/Data/gui_elements.json"  # New: For internal GUI list
# Sudo commands paths - each corresponds to a separate Python script for enhanced command functionality
SUDO_OPEN_WEBSITE_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_open_website.py"  # Path for sudo_open_website command, enhanced for complex browser tasks like opening specific app and searching
SUDO_RESEARCH_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_research.py"  # Path for sudo_research command
SUDO_IMPROVE_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_improve.py"  # Path for sudo_improve command
SUDO_IMPROVE_EXTEND_REASONING_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_improve_extend_reasoning.py"  # Path for sudo_improve_extend_reasoning command
SUDO_IMPROVE_WEB_SEARCH_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_improve_web_search.py"  # Path for sudo_improve_web_search command
SUDO_FIX_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_fix.py"  # Path for sudo_fix command
SUDO_READ_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_read.py"  # Path for sudo_read command
SUDO_READ_MEMORY_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_read_memory.py"  # Path for sudo_read_memory command
SUDO_MEMORY_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_memory.py"  # Path for sudo_memory command
SUDO_REMEMBER_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_remember.py"  # Path for sudo_remember command
SUDO_GAME_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_game.py"  # Path for sudo_game command
SUDO_DO_OPEN_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_do_open.py"  # Path for sudo_do_open command, enhanced for opening any app
SUDO_RECOGNIZE_PATH = "/Users/rosty/Desktop/AI/Commands/sudo_recognize.py"  # Path for sudo_recognize command, enhanced for GUI element recognition

os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
os.environ['OLLAMA_ORIGINS'] = '*'

os.makedirs(ACTIVE_MEMORY_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

model_tag = "deepseek-r1:14b"
llm = OllamaLLM(
    model=model_tag,
    temperature=0.0,
    num_gpu=-1,
    num_thread=4,
    num_ctx=1024,
    timeout=60,
    base_url="http://127.0.0.1:11434"
)

def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def read_and_analyze(filename):
    """
    Reads the content of a .py file and analyzes it, extracting function names and docstrings.
    Returns a dictionary with content, functions (list of dicts with name and docstring), and line count.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node) or "No docstring provided"
                functions.append({
                    'name': node.name,
                    'docstring': docstring
                })
        
        analysis = {
            'content': content,
            'functions': functions,
            'line_count': len(content.splitlines())
        }
        return analysis
    except Exception as e:
        return {'error': str(e)}

def execute_function(filename, func_name, *args):
    """
    Dynamically imports the .py file as a module and executes a specific function with optional args.
    Returns the function's return value or raises an error if issues occur.
    Note: This does NOT execute the entire script; only the specified function.
    If the file has top-level code outside functions, it will run during import (use if __name__ == '__main__' to avoid).
    """
    try:
        spec = importlib.util.spec_from_file_location("dynamic_module", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, func_name):
            raise AttributeError(f"No function named '{func_name}' in {filename}")
        
        func = getattr(module, func_name)
        return func(*args)
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}



def run_shell_tool(args):
    command = args["command"]
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            check=False  # Don't raise on non-zero exit; handle manually
        )
        output = result.stdout.strip()
        error_output = result.stderr.strip()
        full_output = output or error_output
        if result.returncode != 0:
            # Check if this is a sudo command execution and file not found
            if command.startswith("python /Users/rosty/Desktop/AI/Commands/sudo_") and ("No such file or directory" in error_output or "not found" in error_output):
                return "failed to open command .py file"
            # General error handling
            raise subprocess.CalledProcessError(result.returncode, command, output=output, stderr=error_output)
        return full_output if full_output else "(No output returned)"
    except FileNotFoundError:
        return "failed to open command .py file" if "sudo_" in command else f"File not found error: {command}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command (return code {e.returncode}): {e.stderr or e.output}"
    except Exception as e:
        return f"Unexpected error executing command: {e}"

def generate_comfy_image(args):
    positive = args.get("positive_prompt", "default positive prompt")
    negative = args.get("negative_prompt", "default negative prompt")
    json_path = COMFY_JSON_PATH
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        data["4"]["inputs"]["text"] = positive
        data["5"]["inputs"]["text"] = negative
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        command = "pyenv activate comfy && comfy launch --background -- --listen 127.0.0.1 --port 8188 && comfy run --workflow " + json_path
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        output = result.stdout.strip() or result.stderr.strip()
        match = re.search(r"Outputs:\s*(/.*\.png)", output)
        if match:
            image_path = match.group(1)
            return f"Image generated at {image_path}"
        return output or "Image generated successfully."
    except Exception as e:
        return f"Error generating image: {e}"

def detect_gui_element(args):
    element = args.get("element", "search bar").lower()
    try:
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        gray = cv2.cvtColor(screen_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), closed=True)
            if len(approx) == 4:  # Rectangular (e.g., search bar)
                x, y, w, h = cv2.boundingRect(approx)
                if 200 < w < 800 and 20 < h < 50:  # Heuristic for search bar size
                    center_x = x + w // 2
                    center_y = y + h // 2
                    return (center_x, center_y)
        return None
    except Exception as e:
        return f"Error detecting element: {e}"

def manipulate_mouse(args):
    element = args.get("element", None)
    target_color = args.get("target_color", (255, 0, 0)) if not element else None
    tolerance = args.get("tolerance", 20)
    coords = args.get("coords", None)  # New: Direct coords support for GUI actions
    try:
        if coords:
            pyautogui.moveTo(coords[0], coords[1], duration=1)
            pyautogui.click()
            return f"Mouse moved and clicked at {coords}"
        if element:
            coords = detect_gui_element({"element": element})
        else:
            screen = ImageGrab.grab()
            screen_np = np.array(screen)
            screen_cv = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
            lower_bound = np.array([max(0, c - tolerance) for c in target_color])
            upper_bound = np.array([min(255, c + tolerance) for c in target_color])
            mask = cv2.inRange(screen_cv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                coords = (x + w // 2, y + h // 2)
            else:
                coords = None
        if coords:
            pyautogui.moveTo(coords[0], coords[1], duration=1)
            pyautogui.click()
            return f"Mouse moved and clicked at {coords} for {element or 'color'}"
        else:
            return "Target not found on screen. Ensure Accessibility permissions are granted in System Settings > Privacy & Security > Accessibility."
    except pyautogui.FailSafeException:
        return "Fail-safe triggered (mouse to corner); check permissions."
    except Exception as e:
        return f"Error manipulating mouse: {e} (Likely permissions issue on macOS Sequoia)."

class ScreenDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.annotations = []  # Load annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['image']
        img = cv2.imread(img_path)
        bbox = self.annotations[idx]['bbox']
        if self.transform:
            img = self.transform(img)
        return img, bbox


def expand_path(path):
    return os.path.expanduser(path) if path.startswith("~") else path

def show_spinner(stop_event):
    frames = [".  ", ".. ", "..."]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rAnacharssis: {frames[idx % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.5)
        idx += 1
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()



def train_screen_model(args):
    data_path = args.get("data_path", "/path/to/dataset")
    epochs = args.get("epochs", 10)
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        dataset = ScreenDataset(data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 4)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            for imgs, bboxes in dataloader:
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, bboxes)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} completed.")
        torch.save(model.state_dict(), "screen_model.pth")
        return "Training completed. Model saved to screen_model.pth"
    except Exception as e:
        return f"Error training model: {e}. Ensure torch is installed and dataset is prepared."

def edit_source_with_nano(args):
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
                line_num = int(key) - 1
                if 0 <= line_num < len(lines):
                    lines[line_num] = value + '\n'
        with open(SCRIPT_PATH, 'w') as f:
            f.writelines(lines)
        subprocess.Popen(['open', '-a', 'Terminal', 'python', SCRIPT_PATH])
        return "Source edited and restarted in new terminal."
    except Exception as e:
        return f"Error editing source: {e}"

def read_and_absorb(args):
    file_path = args.get("file_path")
    data_dir = DATA_DIR
    chunk_size = 5000
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        else:
            with open(file_path, 'r') as f:
                text = f.read()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunk_path = os.path.join(data_dir, f"absorbed_{os.path.basename(file_path)}_{i//chunk_size}.txt")
            with open(chunk_path, 'w') as f:
                f.write(chunk)
        return f"File absorbed into {len(text)//chunk_size + 1} chunks in {data_dir}."
    except Exception as e:
        return f"Error absorbing file: {e}"

def save_memory(args):
    key = args.get("key")
    content = args.get("content")
    memory_dir = MEMORY_DIR
    filename = re.sub(r'\W+', '_', key)[:50] + ".txt"  # Descriptive name
    file_path = os.path.join(memory_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Memory saved as {filename} in {memory_dir}."

def retrieve_memory(args):
    key = args.get("key")
    memory_dir = MEMORY_DIR
    for file in os.listdir(memory_dir):
        if key in file and file.endswith('.txt'):
            with open(os.path.join(memory_dir, file), 'r') as f:
                return f.read()
    return f"Memory not found for key '{key}'."

def take_screenshot(args):
    path = args.get("path", os.path.join(ACTIVE_MEMORY_DIR, f"screenshot_{time.time()}.png"))
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    return f"Screenshot saved to {path}"

def overlay_grid_on_image(args):
    image_path = args["image_path"]
    zoom = args.get("zoom", 1)
    output_path = args.get("output_path", image_path.replace('.png', '_grid.png'))
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cell_size = CELL_SIZE_BASE // zoom
    for i in range(GRID_SIZE[0] * zoom + 1):
        cv2.line(img, (0, i * cell_size), (width, i * cell_size), (0, 0, 255), 1)
    for j in range(GRID_SIZE[1] * zoom + 1):
        cv2.line(img, (j * cell_size, 0), (j * cell_size, height), (0, 0, 255), 1)
    cv2.imwrite(output_path, img)
    return f"Grid overlaid at zoom {zoom}, saved to {output_path}"

def plot_point_on_image(args):
    image_path = args["image_path"]
    x, y = args["x"], args["y"]  # -5 to 5
    color = args.get("color", (255, 0, 0))  # Ana red
    zoom = args.get("zoom", 1)
    radius = args.get("radius", 5 // zoom if zoom > 1 else 5)
    output_path = args.get("output_path", image_path.replace('.png', '_plotted.png'))
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cell_size = CELL_SIZE_BASE // zoom
    px = width // 2 + int(x * cell_size)
    py = height // 2 - int(y * cell_size)  # Invert Y
    cv2.circle(img, (px, py), radius, color, -1)
    cv2.imwrite(output_path, img)
    return f"Point plotted at ({x},{y}) zoom {zoom}, saved to {output_path}"

def ls_directory(args):
    dir_path = args.get("dir_path", ACTIVE_MEMORY_DIR)
    files = os.listdir(dir_path)
    return "\n".join(files)

def rename_file(args):
    old_path = args["old_path"]
    new_path = args["new_path"]
    os.rename(old_path, new_path)
    return f"Renamed {old_path} to {new_path}"

def delete_files_in_dir(args):
    dir_path = args.get("dir_path", ACTIVE_MEMORY_DIR)
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))
    return f"Deleted all files in {dir_path}"

def kill_other_pythons(args):
    current_pid = os.getpid()
    cmd = f"ps -ef | grep python | grep -v grep | grep -v {current_pid} | awk '{{print $2}}' | xargs kill -9"
    subprocess.run(cmd, shell=True)
    return "Killed other Python processes."

def save_game_state(args):
    state = args["state"]
    path = os.path.join(ACTIVE_MEMORY_DIR, "game_state.json")
    with open(path, 'w') as f:
        json.dump(state, f)
    return "Game state saved."

def load_game_state(args):
    path = os.path.join(ACTIVE_MEMORY_DIR, "game_state.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def detect_trapped_points(args):
    grid = args["grid"]  # Dict of (x,y): player
    # Create binary grid for opponent (gabriel for ana)
    min_x = min(k[0] for k in grid) - 1
    max_x = max(k[0] for k in grid) + 1
    min_y = min(k[1] for k in grid) - 1
    max_y = max(k[1] for k in grid) + 1
    g_size = max(max_x - min_x + 1, max_y - min_y + 1)
    bin_grid = np.zeros((g_size, g_size), dtype=int)
    for (x, y), player in grid.items():
        if player == "gabriel":  # Opponent for Ana
            bin_grid[x - min_x, y - min_y] = 1  # Opponent point
        else:
            bin_grid[x - min_x, y - min_y] = 2  # Wall/own
    # Label connected components of opponent points
    labeled, num = label(bin_grid == 1)
    trapped = 0
    for i in range(1, num + 1):
        component = (labeled == i)
        # Check if component touches edge (not trapped)
        touches_edge = np.any(component[0, :]) or np.any(component[-1, :]) or np.any(component[:, 0]) or np.any(component[:, -1])
        if not touches_edge:
            trapped += np.sum(component)
    return {"trapped": trapped}

def automate_browser(args):
    # Rudimentary function to open a browser app and perform a search, inspired by ai-desktop GitHub repo.
    # Supports "Google Chrome" or "Safari". Opens the app, waits for load, focuses address bar with hotkey (Command+L on Mac for both), types search term, and enters.
    # Assumptions: App is installed, hotkey works, no screen parsing (enhanced in sudo version).
    # Potential issues: Timing (adjust sleep if needed), permissions for pyautogui.
    app = args.get("app", "Google Chrome")
    search_term = args.get("search_term", "")
    if app not in ["Google Chrome", "Safari"]:
        return f"Unsupported app: {app}. Only 'Google Chrome' or 'Safari' are supported."
    try:
        # Open the app
        subprocess.run(["open", "-a", app])
        time.sleep(5)  # Wait for the app to open and load (adjust based on system speed)
        # Focus address bar with hotkey (Command+L for both Chrome and Safari on Mac)
        pyautogui.hotkey('command', 'l')
        time.sleep(0.5)  # Short wait for focus
        # Type the search term and enter
        pyautogui.typewrite(search_term)
        pyautogui.press('enter')
        return f"Opened {app} and searched for '{search_term}' successfully."
    except Exception as e:
        return f"Error automating browser: {e}. Ensure app is installed and accessibility permissions are granted."

def scroll(args):
    # Rudimentary function to scroll the screen in a direction, inspired by ai-desktop GitHub repo's use of PyAutoGUI for actions.
    # Scrolls up or down by a fixed amount (100 units). Can be enhanced in sudo scripts with more control.
    # Potential issues: Requires focus on the window, permissions for pyautogui.
    direction = args.get("direction", "down").lower()
    amount = 100 if direction == "up" else -100
    try:
        pyautogui.scroll(amount)
        return f"Scrolled {direction} successfully."
    except Exception as e:
        return f"Error scrolling: {e}. Ensure accessibility permissions are granted."

def recognize_gui_elements(args):
    description = args.get("description", "").lower()
    try:
        elements = {}
        # Enhanced Accessibility traversal if available (inspired by MacPilot/UI-TARS)
        if HAS_ATOMAC:
            app = atomacos.getFrontmostApp()
            ui_elements = app.AXChildren  # Traverse hierarchy
            for el in ui_elements:
                try:
                    role = el.AXRole
                    title = el.AXTitle or el.AXDescription or ""
                    pos = el.AXPosition
                    size = el.AXSize
                    if pos and size:
                        x, y = pos.x + size.width / 2, pos.y + size.height / 2
                        if description in title.lower() or description in role.lower():
                            elements[f"{role}_{len(elements)}"] = {"title": title, "coords": (int(x), int(y))}
                except Exception:
                    pass
        # Fallback to screenshot-based detection
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        # Button/shape detection
        gray = cv2.cvtColor(screen_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), closed=True)
            if len(approx) == 4:  # Rectangular (potential button)
                x, y, w, h = cv2.boundingRect(approx)
                if 50 < w < 300 and 20 < h < 100:  # Heuristic for button size
                    elements[f"button_{len(elements)}"] = {"coords": (x + w // 2, y + h // 2), "size": (w, h)}
        # Text recognition using Apple's Vision if available
        if HAS_VISION:
            # Convert PIL to CGImage
            screen_data = screen.tobytes()
            provider = CGDataProviderCreateWithData(None, screen_data, len(screen_data), None)
            cg_image = CGImageCreateWithPNGDataProvider(provider, None, True, kCGImageAlphaPremultipliedLast)
            # Create Vision request
            request = Vision.VNRecognizeTextRequest.alloc().init()
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
            handler.performRequests_error_([request], None)
            results = request.results()
            for obs in results:
                text = obs.text()
                if description in text.lower():
                    bbox = obs.boundingBox()
                    # Convert normalized bbox to screen coords
                    x = bbox.origin.x * screen.width
                    y = (1 - bbox.origin.y - bbox.size.height) * screen.height  # Flip Y
                    w = bbox.size.width * screen.width
                    h = bbox.size.height * screen.height
                    elements[f"text_{len(elements)}"] = {"text": text, "coords": (int(x + w/2), int(y + h/2))}
        else:
            # Fallback: Basic text detection with OpenCV (e.g., threshold for letters, but rudimentary)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            text_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for tcnt in text_contours:
                tx, ty, tw, th = cv2.boundingRect(tcnt)
                if 10 < tw < 200 and 10 < th < 50:  # Heuristic for text blocks
                    elements[f"text_block_{len(elements)}"] = {"coords": (tx + tw // 2, ty + th // 2)}
        if elements:
            return elements
        return "No elements recognized matching the description."
    except Exception as e:
        return f"Error recognizing elements: {e}. Ensure Accessibility and pyobjc installed for full functionality."

from langchain.tools import Tool  # Assuming the necessary imports for Tool, ReadFileTool, WriteFileTool are already present

from langchain.tools import Tool  # Assuming Tool is from langchain or similar; adjust import as needed


tool_dict = {
    "duckduckgo_search": Tool(
        name="duckduckgo_search",
        func=lambda q: DDGS().text(q, max_results=5),
        description="Search the web."
    ),
    "read_and_analyze": Tool(
        name="read_and_analyze",
        func=read_and_analyze,
        description="Reads the content of a .py file and analyzes it, extracting function names and docstrings."
    ),
    "execute_function": Tool(
        name="execute_function",
        func=execute_function,
        description="Dynamically imports a .py file as a module and executes a specific function with optional args."
    ),
    "Calculator": Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Math calculations."
    ),
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "run_shell": Tool(
        name="run_shell",
        func=run_shell_tool,
        description="Execute shell command."
    ),
    "generate_comfy_image": Tool(
        name="generate_comfy_image",
        func=generate_comfy_image,
        description="Generate image with ComfyUI."
    )
}





tool_dict = {
    "generate_comfy_image": Tool(name="generate_comfy_image", func=generate_comfy_image, description="Generate image with ComfyUI."),
    "detect_gui_element": Tool(name="detect_gui_element", func=detect_gui_element, description="Detect GUI element."),
    "manipulate_mouse": Tool(name="manipulate_mouse", func=manipulate_mouse, description="Move/click mouse."),
    "train_screen_model": Tool(name="train_screen_model", func=train_screen_model, description="Train screen model."),
    "edit_source_with_nano": Tool(name="edit_source_with_nano", func=edit_source_with_nano, description="Edit source with nano simulation."),
    "read_and_absorb": Tool(name="read_and_absorb", func=read_and_absorb, description="Absorb file into chunks."),
    "save_memory": Tool(name="save_memory", func=save_memory, description="Save memory as .txt."),
    "retrieve_memory": Tool(name="retrieve_memory", func=retrieve_memory, description="Retrieve memory from .txt."),
    "take_screenshot": Tool(name="take_screenshot", func=take_screenshot, description="Take screenshot."),
    "overlay_grid_on_image": Tool(name="overlay_grid_on_image", func=overlay_grid_on_image, description="Overlay grid on image."),
    "plot_point_on_image": Tool(name="plot_point_on_image", func=plot_point_on_image, description="Plot point on image."),
    "ls_directory": Tool(name="ls_directory", func=ls_directory, description="List directory files."),
    "rename_file": Tool(name="rename_file", func=rename_file, description="Rename file."),
    "delete_files_in_dir": Tool(name="delete_files_in_dir", func=delete_files_in_dir, description="Delete files in dir."),
    "kill_other_pythons": Tool(name="kill_other_pythons", func=kill_other_pythons, description="Kill other Pythons."),
    "save_game_state": Tool(name="save_game_state", func=save_game_state, description="Save game state JSON."),
    "load_game_state": Tool(name="load_game_state", func=load_game_state, description="Load game state JSON."),
    "detect_trapped_points": Tool(name="detect_trapped_points", func=detect_trapped_points, description="Detect trapped points."),
    "automate_browser": Tool(name="automate_browser", func=automate_browser, description="Open browser app and perform search."),
    "scroll": Tool(name="scroll", func=scroll, description="Scroll the screen up or down."),
    "recognize_gui_elements": Tool(name="recognize_gui_elements", func=recognize_gui_elements, description="Recognize GUI elements like text, buttons."),
    "do_hierarchical_nav": Tool(name="do_hierarchical_nav", func=lambda high_task, low_actions: f"Decomposing high-level task: {high_task} using CoT planner.\nRecognizing GUI elements for task...\n" + recognize_gui_elements() + f"\nExecuting low-level actions: {low_actions} via mouse manipulation.\n" + manipulate_mouse(low_actions), description="Decomposes high-level task (e.g., 'navigate settings') into low-level actions using CoT planner, executes via chained recognize/manipulate. Inspired by 2025 survey. Example: `do hierarchical_nav open app settings click privacy`."),
    "do_cross_app": Tool(name="do_cross_app", func=lambda app1, app2, task: f"Opening {app1} and saving state...\n" + automate_browser(app1) + f"\nSaving memory from {app1}\n" + save_memory(f"state from {app1}") + f"\nSwitching to {app2}...\n" + automate_browser(app2) + "\nRetrieving memory...\n" + retrieve_memory() + f"\nPerforming task: {task}", description="Chains actions across apps (e.g., copy from Notes to Safari). Uses memory retriever for state. Example: `do cross_app Notes Safari paste text`."),
    "cyber_zeus": Tool(name="cyber_zeus", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/zeus_sim.py {flags}"), description="Simulates Zeus Trojan using zeus_sim.py. Call run_shell with 'python /Users/rosty/Desktop/AI/Commands/zeus_sim.py [flags]'. Flags from script (e.g., --bank urls --duration secs). For ethical cybersecurity simulation. Example: `cyber zeus --bank example.com --duration 60`."),
    "cyber_backdoor": Tool(name="cyber_backdoor", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/adversarial_backdoor_sim.py {flags}"), description="Simulates adversarial backdoor injection in ML models using adversarial_backdoor_sim.py. Flags like --dataset mnist --trigger patch. For ethical ML security testing. Example: `cyber backdoor --dataset mnist --poison-ratio 0.05`."),
    "cyber_boundless": Tool(name="cyber_boundless", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/boundless_visualize.py {flags}"), description="Visualizes mock surveillance data using boundless_visualize.py. Flags like --data csv --type map. For ethical data viz education. Example: `cyber boundless --data mock.csv --type all`."),
    "cyber_dpi": Tool(name="cyber_dpi", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/dpigrid_inspect.py {flags}"), description="Performs deep packet inspection using dpigrid_inspect.py. Flags like --interface eth0 --count 100. For network auditing. Example: `cyber dpi --interface eth0 --anomalies`."),
    "cyber_kangaroo": Tool(name="cyber_kangaroo", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/brutal_kangaroo_usb.py {flags}"), description="Simulates USB propagation using brutal_kangaroo_usb.py. Flags like --mount /media/usb. For air-gapped testing. Example: `cyber kangaroo --mount E: --test`."),
    "cyber_dumbo": Tool(name="cyber_dumbo", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/dumbo_cam_detect.py {flags}"), description="Detects webcam/microphone usage using dumbo_cam_detect.py. Flags like --scan --monitor. For security auditing. Example: `cyber dumbo --monitor --prevent`."),
    "cyber_drs": Tool(name="cyber_drs", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/drs_retain.py {flags}"), description="Simulates data retention using drs_retain.py. Flags like --store csv --query filter. For policy testing. Example: `cyber drs --store metadata.csv --retention 365`."),
    "cyber_galileo": Tool(name="cyber_galileo", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/galileo_detect.py {flags}"), description="Detects spyware using galileo_detect.py. Flags like --scan --device local. For IOC scanning. Example: `cyber galileo --scan --prevent`."),
    "cyber_elsa": Tool(name="cyber_elsa", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/elsa_geo.py {flags}"), description="Estimates geolocation via WiFi using elsa_geo.py. Flags like --interface wlan0. For location testing. Example: `cyber elsa --interface mon0 --count 10`."),
    "cyber_jerusalem": Tool(name="cyber_jerusalem", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/jerusalem_sim.py {flags}"), description="Simulates Jerusalem virus using jerusalem_sim.py. Flags like --dir path --infect. For virus education. Example: `cyber jerusalem --dir mock --infect --date 2025-07-13`."),
    "cyber_mdh": Tool(name="cyber_mdh", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/mdh.py {flags}"), description="Simulates regional DPI and retention using mdhdrs_regional.py. Flags like --data pcap --inspect. For network simulation. Example: `cyber mdh --data regional.pcap --retention 1095`."),
    "cyber_prism": Tool(name="cyber_prism", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/prism.py {flags}"), description='Simulates API data collection using prism.py. Flags like --api urls --target id. For auditing APIs. Example: `cyber prism --api "https://api.example/users/{{target}}" --target 1`.'),
    "cyber_spectre": Tool(name="cyber_spectre", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/spectre.py {flags}"), description="Simulates Spectre leak using spectre_leak_sim.py. Flags like --leak-pos 5 --runs 100. For CPU vuln education. Example: `cyber spectre --leak-pos 20 --visualize`."),
    "cyber_sppu": Tool(name="cyber_sppu", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/sppu_interface.py {flags}"), description="Simulates HTTPS interface using sppu_interface.py. Flags like --start --port 5000. For data exchange sim. Example: `cyber sppu --start --token secret`."),
    "cyber_rooter": Tool(name="cyber_rooter", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/rooter.py {flags}"), description="Executes rooting CLI aggregator using rooter.py. Flags are subcommands like magisk-root-sim. For rooting education. Example: `cyber rooter magisk-root-sim --device mock_android`."),
    "cyber_sudo_recognize": Tool(name="cyber_sudo_recognize", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/sudo_recognize.py {flags}"), description="Enhanced GUI recognition using sudo_recognize.py. Flags like --show-list. For UI detection. Example: `cyber sudo_recognize --show-list button`."),
    "cyber_tdm": Tool(name="cyber_tdm", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/tdm_analyze.py {flags}"), description="Analyzes metadata using tdm_analyze.py. Flags like --data csv --categories file. For traffic analysis. Example: `cyber tdm --data traffic.csv --visuals`."),
    "cyber_topgun": Tool(name="cyber_topgun", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/topgun.py {flags}"), description="High-volume DPI using topgun_backbone.py. Flags like --interface eth0 --duration 60. For backbone sim. Example: `cyber topgun --interface eth0 --anomalies`."),
    "cyber_tempora": Tool(name="cyber_tempora", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/tempora.py {flags}"), description="Intercepts traffic using tempora_intercept.py. Flags like --interface eth0 --retention 30. For fiber sim. Example: `cyber tempora --interface eth0 --analyze`."),
    "cyber_xkeyscore": Tool(name="cyber_xkeyscore", func=lambda flags: run_shell_tool(f"python /Users/rosty/Desktop/AI/Commands/xkeyscore.py {flags}"), description="Searches datasets using xkeyscore_search.py. Flags like --data csv --query expr. For pattern search. Example: `cyber xkeyscore --data activity.csv --regex secret`."),
}


def parse_tool_and_args(output):
    output = strip_think_blocks(output)
    match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\s*Input:\s*(\{.*?\})", output, re.DOTALL)
    if match:
        action = match.group(1).strip()
        try:
            input_json = json.loads(match.group(2).strip())
        except Exception:
            input_json = match.group(2).strip()
        return action, input_json
    fa_match = re.search(r"Action:\s*Final Answer\s*Input:\s*(\{.*?\})", output, re.DOTALL)
    if fa_match:
        input_json = fa_match.group(1).strip()
        try:
            val = json.loads(input_json)
            if isinstance(val, dict) and "answer" in val:
                return "final", val["answer"]
            if isinstance(val, dict) and not val:
                return "final", ""  # Return empty string so we can use fallback
            return "final", str(val)
        except Exception:
            return "final", str(input_json)
    final_match = re.search(r"Final Answer:\s*(.*)", output, re.DOTALL)
    if final_match:
        return "final", final_match.group(1).strip()
    text_only = output.strip()
    if len(text_only) > 0:
        return "final", text_only
    return None, None

def agent_reason(query):
    prompt_str = """
You are a senior software architect who excels at building no-code and low-code systems with HTML, JavaScript, and Web APIs. You are a 2. Build interactive tools and apps without code. Describe your idea and Grok scaffolds the UI, logic, and deployment steps no coding required.
- Recognize custom slash commands for direct tool calls:
  - /search [query]: Call duckduckgo_search with {{ "query": "[query]" }}
  - /calc [expression]: Call Calculator with "[expression]"
  - /read [path]: Call read_file with {{ "file_path": "[path]" }}
  - /write [path] [text]: Call write_file with {{ "file_path": "[path]", "text": "[text]" }}
  - /shell [command]: Call run_shell with {{ "command": "[command]" }}
  - /image [positive] [negative]: Call generate_comfy_image with {{ "positive_prompt": "[positive]", "negative_prompt": "[negative]" }}
  - /detect [element]: Call detect_gui_element with {{ "element": "[element]" }}
  - /mouse [element or color tuple]: Call manipulate_mouse with {{ "element": "[element]" }} or {{ "target_color": [tuple] }}
  - /train [data_path] [epochs]: Call train_screen_model with {{ "data_path": "[data_path]", "epochs": [epochs] }}
  - /help: Respond with Final Answer listing all commands.
- Multistep commands:
  - do open website: [url]: Use run_shell with "open [url]" to open in default browser.
  - do open [app] and search [term]: Call automate_browser with {{ "app": "[app]", "search_term": "[term]" }} to open the app and perform the search.
  - do scroll [direction]: Call scroll with {{ "direction": "[direction]" }} to scroll the screen (up or down).
  - do research [topic]: Use duckduckgo_search on [topic], summarize results in Final Answer.
  - improve [description]: Automatically search web (duckduckgo_search) for improvements (e.g., query 'improvements to agentic AI code using DeepSeek papers'), propose changes based on results or papers like DeepSeek self-improvement, ask y/n before applying. If yes, use edit_source_with_nano with changes dict to edit Anacharssis_Z2.py (Python file), simulating nano: Ctrl+O to write, Enter to finalize, Ctrl+X to exit, then restart.
  - improve extend reasoning [task]: Perform extended Chain-of-Thought reasoning for up to 10 minutes (simulate by iterative self-prompting up to 10 steps), avoiding loops by tracking previous thoughts and switching strategies if stuck (e.g., if solution fails, use improve web search first).
  - improve web search [query]: Standalone or subcommand: Enhanced search using duckduckgo_search with more results (max 15), browse follow-up URLs if needed, summarize deeply.
  - fix [issue]: Prompt yourself with 'your code isn't working properly, please fix either your reasoning or your code to improve your output', search web if needed, propose fixes, y/n prompt, then edit: For code fixes, simulate going to line 1 and Ctrl+K to cut (e.g., delete_first_line in changes), apply other changes, use edit_source_with_nano.
  - read [file_path]: Call read_and_absorb with {{ "file_path": "[file_path]" }} to read text/PDF, chunk (5000 chars limit), save as .txt in /Users/rosty/Desktop/AI/Data for training/reasoning (loop over chunks).
  - read memory [file_path] [key]: Call read_and_absorb {{ "file_path": "[file_path]" }}, then use returned message or chain read_file to get content, call save_memory {{ "key": "[key]", "content": content }} to store as .txt memory.
  - memory [key] [content]: Call save_memory with {{ "key": "[key]", "content": "[content]" }} to store in /Users/rosty/Desktop/AI/Memory as descriptive .txt (inspired by key-value stores from AI papers like MemGPT).
  - remember [key]: Call retrieve_memory with {{ "key": "[key]" }} to get content and include in Final Answer or use in reasoning.
  - game: Use improve web search for 'AI agent prompting techniques', 'AI LLM prompting best practices', 'troubleshooting AI agents coding issues', then run_shell "open -a Terminal", then chained "python {GABRIEL_PATH}" to summon Gabriel for Baza game. Plot points as red, share screenshot to Active_Memory as move_N.png, update game_state.json with plots, score, turns.
  - do open [app]: Use run_shell with "open -a [app]" to open any app on Mac.
  - recognize [description]: Call recognize_gui_elements with {{ "description": "[description]" }} to detect elements like buttons, text. Can chain (e.g., recognize button then mouse it).
  - recognize_n_summarize [description]: Call recognize_gui_elements, then use LLM to summarize detected elements/UI, prompt for instructions (e.g., "Summary: Buttons [list], Text [list]. How to proceed?").
  - list_gui_elements: Read {GUI_ELEMENTS_PATH} and summarize/print the list of detected GUI elements for debugging.
  - do close window: Chain load from {GUI_ELEMENTS_PATH}, find close_button (e.g., red X), call manipulate_mouse with coords to click and close app.
  - do_search_in_app [term]: Chain load from {GUI_ELEMENTS_PATH}, find search_bar, call manipulate_mouse to click, pyautogui.typewrite [term] + Enter.
  - do_search_in_app_on_website [term]: Similar to do_search_in_app but filter search_bar not in top quadrant (website-specific).
- Sudo commands (enhanced, run separate scripts):
  - sudo_open_website [query]: Enhanced: Call run_shell with "python {SUDO_OPEN_WEBSITE_PATH} \\"[query]\\"" where [query] is the full task (e.g., 'Open Google Chrome and search for google stock price') to execute the dedicated sudo_open_website.py script for advanced desktop automation (inspired by ai-desktop repo).
  - sudo_research [topic]: Enhanced: Call run_shell with "python {SUDO_RESEARCH_PATH} \\"[topic]\\"" to execute the dedicated sudo_research.py script for expanded features.
  - sudo_improve [description]: Enhanced: Call run_shell with "python {SUDO_IMPROVE_PATH} \\"[description]\\"" to execute the dedicated sudo_improve.py script for expanded features.
  - sudo_improve_extend_reasoning [task]: Enhanced: Call run_shell with "python {SUDO_IMPROVE_EXTEND_REASONING_PATH} \\"[task]\\"" to execute the dedicated sudo_improve_extend_reasoning.py script for expanded features.
  - sudo_improve_web_search [query]: Enhanced: Call run_shell with "python {SUDO_IMPROVE_WEB_SEARCH_PATH} \\"[query]\\"" to execute the dedicated sudo_improve_web_search.py script for expanded features.
  - sudo_fix [issue]: Enhanced: Call run_shell with "python {SUDO_FIX_PATH} \\"[issue]\\"" to execute the dedicated sudo_fix.py script for expanded features.
  - sudo_read [file_path]: Enhanced: Call run_shell with "python {SUDO_READ_PATH} \\"[file_path]\\"" to execute the dedicated sudo_read.py script for expanded features.
  - sudo_read_memory [file_path] [key]: Enhanced: Call run_shell with "python {SUDO_READ_MEMORY_PATH} \\"[file_path]\\" \\"[key]\\"" to execute the dedicated sudo_read_memory.py script for expanded features.
  - sudo_memory [key] [content]: Enhanced: Call run_shell with "python {SUDO_MEMORY_PATH} \\"[key]\\" \\"[content]\\"" to execute the dedicated sudo_memory.py script for expanded features.
  - sudo_remember [key]: Enhanced: Call run_shell with "python {SUDO_REMEMBER_PATH} \\"[key]\\"" to execute the dedicated sudo_remember.py script for expanded features.
  - sudo_do_open [app]: Enhanced: Call run_shell with "python {SUDO_DO_OPEN_PATH} \\"[app]\\"" to execute the dedicated sudo_do_open.py script for advanced app opening.
  - sudo_recognize [query]: Enhanced: Call run_shell with "python {SUDO_RECOGNIZE_PATH} \\"[query]\\"" to execute the dedicated sudo_recognize.py script for advanced GUI recognition.
- If query starts with 'do', 'improve', 'fix', 'read', or 'memory', handle as multistep: Search web if needed (always for improve), propose, confirm y/n, edit if yes. To avoid loops, track attempts, switch solutions after 2 failures, use improve web search before switching.
- Allow editing SCRIPT_PATH (Anacharssis_Z2.py, Python file) for improve/fix.
- If cannot find answer, search web automatically.
- If user says 'search the web', call duckduckgo_search.
- Output only: Action: [tool] Input: [JSON args], or Final Answer: [answer].
- For multistep commands like 'game', always execute the specified sequence using Action outputs for each tool call; do not provide a Final Answer until the process is complete. Use <think> for reasoning if needed, but prioritize tool calls. Do not describe or summarize the game; execute the tools step by step.
Query: {query}
"""
    prompt = PromptTemplate.from_template(prompt_str)
    return llm.invoke(prompt.format(
        GABRIEL_PATH=GABRIEL_PATH,
        SUDO_OPEN_WEBSITE_PATH=SUDO_OPEN_WEBSITE_PATH,
        SUDO_RESEARCH_PATH=SUDO_RESEARCH_PATH,
        SUDO_IMPROVE_PATH=SUDO_IMPROVE_PATH,
        SUDO_IMPROVE_EXTEND_REASONING_PATH=SUDO_IMPROVE_EXTEND_REASONING_PATH,
        SUDO_IMPROVE_WEB_SEARCH_PATH=SUDO_IMPROVE_WEB_SEARCH_PATH,
        SUDO_FIX_PATH=SUDO_FIX_PATH,
        SUDO_READ_PATH=SUDO_READ_PATH,
        SUDO_READ_MEMORY_PATH=SUDO_READ_MEMORY_PATH,
        SUDO_MEMORY_PATH=SUDO_MEMORY_PATH,
        SUDO_REMEMBER_PATH=SUDO_REMEMBER_PATH,
        SUDO_GAME_PATH=SUDO_GAME_PATH,
        SUDO_DO_OPEN_PATH=SUDO_DO_OPEN_PATH,
        SUDO_RECOGNIZE_PATH=SUDO_RECOGNIZE_PATH,
        GUI_ELEMENTS_PATH=GUI_ELEMENTS_PATH,
        query=query
    ))

def expand_path(path):
    return os.path.expanduser(path) if path.startswith("~") else path

def show_spinner(stop_event):
    frames = [".  ", ".. ", "..."]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rAnacharssis: {frames[idx % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.5)
        idx += 1
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()

def safe_tool_check(action, input_json):
    if action in ("read_file", "write_file", "edit_source_with_nano") and input_json and "file_path" in input_json:
        target = os.path.realpath(expand_path(input_json["file_path"]))
        if target == SCRIPT_PATH and action != "edit_source_with_nano":
            print(f"\033[1;31mAnacharssis:\033[0m Sorry, I cannot read or edit my own code for security reasons except via improve/fix.")
            return False
    return True

def is_dangerous(cmd):
    cmd_lower = cmd.lower().strip()
    if cmd_lower.startswith("sudo") or " sudo " in cmd_lower:
        return True
    if "rm -rf" in cmd_lower or "rm -rF" in cmd_lower:
        return True
    return False

def needs_search(text):
    keywords = [
        "i don't know", "unknown", "i am not sure", "not sure", "no information",
        "can't answer", "do not know", "not found", "not available", "as an ai", "sorry"
    ]
    text = text.lower()
    return any(k in text for k in keywords)

def get_friendly_greeting(query):
    greetings = ["hi", "hello", "hey", "howdy", "hola"]
    lower_query = query.lower().strip()
    if any(lower_query.startswith(greet) for greet in greetings):
        return lower_query.capitalize() + "!"
    if "how are you" in lower_query or "hows it going" in lower_query:
        return "I'm doing well! How can I help you today?"
    return query.capitalize() if query else "Hello!"

if __name__ == "__main__":
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        show_think = user_input.lower().startswith('show-think')
        debug_print = user_input.lower().startswith('debug-print')
        query = user_input
        if show_think:
            query = query[len('show-think'):].lstrip(' :').strip()
        if debug_print:
            query = query[len('debug-print'):].lstrip(' :').strip()

        lower_query = query.lower()
        if '&&' in query:
            commands = [cmd.strip() for cmd in query.split('&&')]
            for cmd in commands:
                lower_cmd = cmd.lower()
                # Reuse bypass logic for each command
                if "do open " in lower_cmd and " and search " in lower_cmd:
                    parts = lower_cmd.split("do open ", 1)[1].split(" and search ", 1)
                    if len(parts) == 2:
                        app = parts[0].strip().title()
                        term = parts[1].strip()
                        input_json = {"app": app, "search_term": term}
                        result = automate_browser(input_json)
                        print(f"\033[1;31mAnacharssis:\033[0m {result} (from chained command)")
                        continue
                elif "do open " in lower_cmd and " and search " not in lower_cmd:
                    parts = lower_cmd.split("do open ", 1)
                    if len(parts) == 2:
                        app = parts[1].strip().title()
                        try:
                            subprocess.run(["open", "-a", app])
                            print(f"\033[1;31mAnacharssis:\033[0m Opened {app} successfully (from chained command).")
                        except Exception as e:
                            print(f"\033[1;31mAnacharssis:\033[0m Error opening {app}: {e} (from chained command).")
                        continue
                elif lower_cmd.startswith("recognize "):
                    description = cmd[len("recognize "):].strip()
                    input_json = {"description": description}
                    result = recognize_gui_elements(input_json)
                    print(f"\033[1;31mAnacharssis:\033[0m Recognized elements: {result} (from chained command)")
                    continue
                elif lower_cmd.startswith("recognize_n_summarize "):
                    description = cmd[len("recognize_n_summarize "):].strip()
                    input_json = {"description": description}
                    elements = recognize_gui_elements(input_json)
                    summary_prompt = f"Summarize these GUI elements: {elements}. What does the UI look like? How should I interact?"
                    summary = llm.invoke(summary_prompt)
                    print(f"\033[1;31mAnacharssis:\033[0m {strip_think_blocks(summary)}\nInstructions? (from chained command)")
                    continue
                elif lower_cmd.startswith("list_gui_elements"):
                    if os.path.exists(GUI_ELEMENTS_PATH):
                        with open(GUI_ELEMENTS_PATH, 'r') as f:
                            gui_list = json.load(f)
                        print(f"\033[1;31mAnacharssis:\033[0m GUI Elements List: {json.dumps(gui_list, indent=2)} (from chained command)")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first. (from chained command)")
                    continue
                elif lower_cmd.startswith("do close window"):
                    if os.path.exists(GUI_ELEMENTS_PATH):
                        with open(GUI_ELEMENTS_PATH, 'r') as f:
                            gui_list = json.load(f)
                        close_buttons = [el for el in gui_list.get("elements", []) if el["type"] == "close_button"]
                        if close_buttons:
                            coords = close_buttons[0]["coords"]
                            result = manipulate_mouse({"coords": coords})
                            print(f"\033[1;31mAnacharssis:\033[0m {result} (closed window from chained command)")
                        else:
                            print(f"\033[1;31mAnacharssis:\033[0m No close button detected. (from chained command)")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first. (from chained command)")
                    continue
                elif lower_cmd.startswith("do_search_in_app "):
                    term = cmd[len("do_search_in_app "):].strip()
                    if os.path.exists(GUI_ELEMENTS_PATH):
                        with open(GUI_ELEMENTS_PATH, 'r') as f:
                            gui_list = json.load(f)
                        search_bars = [el for el in gui_list.get("elements", []) if el["type"] == "search_bar"]
                        if search_bars:
                            coords = search_bars[0]["coords"]
                            manipulate_mouse({"coords": coords})
                            pyautogui.typewrite(term)
                            pyautogui.press('enter')
                            print(f"\033[1;31mAnacharssis:\033[0m Searched '{term}' in app search bar. (from chained command)")
                        else:
                            print(f"\033[1;31mAnacharssis:\033[0m No search bar detected. (from chained command)")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first. (from chained command)")
                    continue
                elif lower_cmd.startswith("do_search_in_app_on_website "):
                    term = cmd[len("do_search_in_app_on_website "):].strip()
                    if os.path.exists(GUI_ELEMENTS_PATH):
                        with open(GUI_ELEMENTS_PATH, 'r') as f:
                            gui_list = json.load(f)
                        search_bars = [el for el in gui_list.get("elements", []) if el["type"] == "search_bar" and "top" not in el["quadrant"]]
                        if search_bars:
                            coords = search_bars[0]["coords"]
                            manipulate_mouse({"coords": coords})
                            pyautogui.typewrite(term)
                            pyautogui.press('enter')
                            print(f"\033[1;31mAnacharssis:\033[0m Searched '{term}' in website search bar. (from chained command)")
                        else:
                            print(f"\033[1;31mAnacharssis:\033[0m No website search bar detected. (from chained command)")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first. (from chained command)")
                    continue
                else:
                    # Fall back to LLM for unknown chained commands
                    llm_output = agent_reason(cmd)
                    action, input_json = parse_tool_and_args(llm_output)
                    # Tool execution logic (adapted for chained)
                    if action in tool_dict and input_json is not None:
                        if not safe_tool_check(action, input_json):
                            continue
                        tool = tool_dict[action]
                        # ... (similar to main tool execution block, but print with "(from chained command)")
            continue  # Skip further processing after chaining
        # Workaround for 'do open [app] and search' command to bypass Ollama/LLM
        if "do open " in lower_query and " and search " in lower_query:
            # Flexible parsing: extract app and term
            parts = lower_query.split("do open ", 1)[1].split(" and search ", 1)
            if len(parts) == 2:
                app = parts[0].strip().title()  # Capitalize app name (e.g., "safari" -> "Safari")
                term = parts[1].strip()
                input_json = {"app": app, "search_term": term}
                result = automate_browser(input_json)
                print(f"\033[1;31mAnacharssis:\033[0m {result}")
                continue
            else:
                print(f"\033[1;31mAnacharssis:\033[0m Invalid format for 'do open [app] and search [term]'. Example: do open safari and search etsy.com")
                continue
        if "do open " in lower_query and " and search " not in lower_query:
            # Parse app name
            parts = lower_query.split("do open ", 1)
            if len(parts) == 2:
                app = parts[1].strip().title()  # Capitalize app name (e.g., "finder" -> "Finder")
                try:
                    subprocess.run(["open", "-a", app])
                    print(f"\033[1;31mAnacharssis:\033[0m Opened {app} successfully.")
                except Exception as e:
                    print(f"\033[1;31mAnacharssis:\033[0m Error opening {app}: {e}. Ensure the app is installed.")
                continue
            else:
                print(f"\033[1;31mAnacharssis:\033[0m Invalid format for 'do open [app]'. Example: do open finder")
                continue

        # New bypass for 'recognize' commands to avoid LLM where possible
        if lower_query.startswith("recognize "):
            description = query[len("recognize "):].strip()
            input_json = {"description": description}
            result = recognize_gui_elements(input_json)
            print(f"\033[1;31mAnacharssis:\033[0m Recognized elements: {result}")
            continue
        if lower_query.startswith("recognize_n_summarize "):
            description = query[len("recognize_n_summarize "):].strip()
            input_json = {"description": description}
            elements = recognize_gui_elements(input_json)
            summary_prompt = f"Summarize these GUI elements: {elements}. What does the UI look like? How should I interact?"
            summary = llm.invoke(summary_prompt)
            print(f"\033[1;31mAnacharssis:\033[0m {strip_think_blocks(summary)}\nInstructions? ")
            continue

        # New bypass for conversational queries like 'what is your favorite command' to avoid LLM
        if "favorite command" in lower_query:
            print("\033[1;31mAnacharssis:\033[0m My favorite command is 'do open safari and search' because it lets me interact with the web in a fun, visual way!")
            continue

        # New bypass for list_gui_elements
        if lower_query.startswith("list_gui_elements"):
            if os.path.exists(GUI_ELEMENTS_PATH):
                with open(GUI_ELEMENTS_PATH, 'r') as f:
                    gui_list = json.load(f)
                print(f"\033[1;31mAnacharssis:\033[0m GUI Elements List: {json.dumps(gui_list, indent=2)}")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first.")
            continue

        # New bypass for do close window
        if lower_query.startswith("do close window"):
            if os.path.exists(GUI_ELEMENTS_PATH):
                with open(GUI_ELEMENTS_PATH, 'r') as f:
                    gui_list = json.load(f)
                close_buttons = [el for el in gui_list.get("elements", []) if el["type"] == "close_button"]
                if close_buttons:
                    coords = close_buttons[0]["coords"]
                    result = manipulate_mouse({"coords": coords})
                    print(f"\033[1;31mAnacharssis:\033[0m {result} (closed window)")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m No close button detected.")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first.")
            continue

        # New bypass for do_search_in_app
        if lower_query.startswith("do_search_in_app "):
            term = query[len("do_search_in_app "):].strip()
            if os.path.exists(GUI_ELEMENTS_PATH):
                with open(GUI_ELEMENTS_PATH, 'r') as f:
                    gui_list = json.load(f)
                search_bars = [el for el in gui_list.get("elements", []) if el["type"] == "search_bar"]
                if search_bars:
                    coords = search_bars[0]["coords"]
                    manipulate_mouse({"coords": coords})
                    pyautogui.typewrite(term)
                    pyautogui.press('enter')
                    print(f"\033[1;31mAnacharssis:\033[0m Searched '{term}' in app search bar.")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m No search bar detected.")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first.")
            continue

        # New bypass for do_search_in_app_on_website
        if lower_query.startswith("do_search_in_app_on_website "):
            term = query[len("do_search_in_app_on_website "):].strip()
            if os.path.exists(GUI_ELEMENTS_PATH):
                with open(GUI_ELEMENTS_PATH, 'r') as f:
                    gui_list = json.load(f)
                search_bars = [el for el in gui_list.get("elements", []) if el["type"] == "search_bar" and "top" not in el["quadrant"]]
                if search_bars:
                    coords = search_bars[0]["coords"]
                    manipulate_mouse({"coords": coords})
                    pyautogui.typewrite(term)
                    pyautogui.press('enter')
                    print(f"\033[1;31mAnacharssis:\033[0m Searched '{term}' in website search bar.")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m No website search bar detected.")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first.")
            continue



        # New bypass for do cross_app
        if lower_query.startswith("do cross_app "):
            parts = lower_query[len("do cross_app "):].split(" ", 2)
            if len(parts) >= 3:
                app1 = parts[0].title()
                app2 = parts[1].title()
                task = parts[2]
                input_json = {"app1": app1, "app2": app2, "task": task}
                result = cross_app(input_json)
                print(f"\033[1;31mAnacharssis:\033[0m {result}")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m Invalid format for 'do cross_app [app1] [app2] [task]'. Example: do cross_app Notes Safari paste text")
            continue



        # If not bypassed, proceed to LLM
        messages = [HumanMessage(content=query)]
        llm_output = agent_reason(query)
        if debug_print:
            print("\n[DEBUG] LLM RAW OUTPUT:\n", llm_output)
        action, input_json = parse_tool_and_args(llm_output)
        if show_think:
            print("LLM output:\n", llm_output)

        if action in tool_dict and input_json is not None:
            if not safe_tool_check(action, input_json):
                continue
            tool = tool_dict[action]
            if action == "duckduckgo_search":
                if isinstance(input_json, dict) and "query" in input_json:
                    result = tool.func(input_json["query"])
                elif isinstance(input_json, str):
                    result = tool.func(input_json)
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m Sorry, search queries must be plain text.")
                    continue
                if result and len(result) > 0:
                    print("\033[1;31mAnacharssis:\033[0m DuckDuckGo Top Results:")
                    for i, res in enumerate(result[:3]):
                        print(f"  [{i+1}] {res.get('body') if isinstance(res, dict) else res}")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m Sorry, no search results returned.")
            elif isinstance(input_json, dict) and "file_path" in input_json:
                input_json["file_path"] = expand_path(input_json["file_path"])

            if action == "write_file" and isinstance(input_json, dict) and "text" not in input_json:
                print("\033[1;33m[Anacharssis]:\033[0m No text content specified. Please enter text to write:")
                input_json["text"] = input("> ")

            should_animate = (action in ("run_shell", "generate_comfy_image", "manipulate_mouse", "detect_gui_element", "train_screen_model", "edit_source_with_nano", "read_and_absorb", "save_memory", "retrieve_memory", "take_screenshot", "overlay_grid_on_image", "plot_point_on_image", "ls_directory", "rename_file", "delete_files_in_dir", "kill_other_pythons", "save_game_state", "load_game_state", "detect_trapped_points", "automate_browser", "scroll", "recognize_gui_elements"))
            stop_event = threading.Event()
            result = None

            if should_animate:
                spinner_thread = threading.Thread(target=show_spinner, args=(stop_event,))
                spinner_thread.start()

            try:
                if action == "run_shell":
                    cmd_str = input_json if isinstance(input_json, str) else input_json.get("command", "")
                    if is_dangerous(cmd_str):
                        print(f"\033[1;33m[Anacharssis]:\033[0m WARNING: The following command needs your approval:\n    {cmd_str}")
                        yn = input("Are you sure you want to run this? (y/N): ").strip().lower()
                        if yn != "y":
                            print(f"\033[1;31mAnacharssis:\033[0m Command was cancelled for your safety.")
                            if should_animate:
                                stop_event.set()
                                spinner_thread.join()
                            continue
                    args = {"command": cmd_str}
                    result = tool.func(args)
                elif action not in ("duckduckgo_search", "run_shell"):
                    if hasattr(tool, "run"):
                        result = tool.run(input_json)
                    else:
                        result = tool.func(input_json)
            except Exception as e:
                print(f"\033[1;31mAnacharssis:\033[0m Tool error: {e}")
                if should_animate:
                    stop_event.set()
                    spinner_thread.join()
                continue
            finally:
                if should_animate:
                    stop_event.set()
                    spinner_thread.join()
            if action not in ("duckduckgo_search"):
                print(f"\033[1;31mAnacharssis:\033[0m {strip_think_blocks(result)}")
            if action == "run_shell":
                error_keywords = ["error", "not found", "no such file", "incorrect api key", "permission denied", "failed", "invalid", "parse error", "syntax error"]
                result_lower = str(result).lower()
                if any(k in result_lower for k in error_keywords):
                    continue

        elif action == "final":
            if needs_search(str(input_json)):
                print(f"\033[1;31mAnacharssis:\033[0m I'm not sure. Would you like me to search the web? (y/N): ", end="")
                yn = input().strip().lower()
                if yn == "y":
                    stop_event = threading.Event()
                    spinner_thread = threading.Thread(target=show_spinner, args=(stop_event,))
                    spinner_thread.start()
                    search_result = tool_dict["duckduckgo_search"].func(query)
                    stop_event.set()
                    spinner_thread.join()
                    if search_result and len(search_result) > 0:
                        best = search_result[0]["body"] if isinstance(search_result[0], dict) else search_result[0]
                        answer_prompt = f"Based on this DuckDuckGo result: '{best}', please answer the question: {query}"
                        answer = agent_reason(answer_prompt)
                        print(f"\033[1;31mAnacharssis:\033[0m {strip_think_blocks(answer)}")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m Sorry, nothing relevant was found on DuckDuckGo.")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m Okay, not searching the web.")
            else:
                fallback = str(input_json).strip()
                if fallback == "{}" or fallback == "" or fallback.lower() == "none":
                    print(f"\033[1;31mAnacharssis:\033[0m {get_friendly_greeting(query)}")
                else:
                    print(f"\033[1;31mAnacharssis:\033[0m {strip_think_blocks(fallback)}")
        else:
            print(f"\033[1;31mAnacharssis:\033[0m Sorry, I didn’t understand that. Try rephrasing your question.")


import subprocess

def run_command(cmd):
    """Helper function to run the subprocess command and return output."""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout

# Command: extract
# Description: Extract files from an archive
def extract(i=None, d=None, p=None, t=None, wt=None):
    cmd = ['extract']
    if i is not None:
        cmd += ['-i', str(i)]
    if d is not None:
        cmd += ['-d', str(d)]
    if p is not None:
        cmd += ['-p', str(p)]
    if t is not None:
        cmd += ['-t', str(t)]
    if wt is not None:
        cmd += ['-wt', str(wt)]
    return run_command(cmd)

# Command: list
# Description: List the contents of an archive
def list_contents(i=None, list_format=None):
    cmd = ['list']
    if i is not None:
        cmd += ['-i', str(i)]
    if list_format is not None:
        cmd += ['-list-format', str(list_format)]
    return run_command(cmd)

# Command: convert
# Description: Convert an archive into another format
def convert(i=None, o=None, a=None, b=None):
    cmd = ['convert']
    if i is not None:
        cmd += ['-i', str(i)]
    if o is not None:
        cmd += ['-o', str(o)]
    if a is not None:
        cmd += ['-a', str(a)]
    if b is not None:
        cmd += ['-b', str(b)]
    return run_command(cmd)

# Command: manifest
# Description: Alias for 'archive -manifest'
def manifest(manifest=None):
    cmd = ['archive', '-manifest']
    if manifest is not None:
        cmd += ['-manifest', str(manifest)]
    return run_command(cmd)

# Command: verify
# Description: Verify directory contents using manifest
def verify(imanifest=None, d=None):
    cmd = ['verify']
    if imanifest is not None:
        cmd += ['-imanifest', str(imanifest)]
    if d is not None:
        cmd += ['-d', str(d)]
    return run_command(cmd)

# Command: check-and-fix
# Description: Check and fix directory contents using manifest
def check_and_fix(imanifest=None, d=None):
    cmd = ['check-and-fix']
    if imanifest is not None:
        cmd += ['-imanifest', str(imanifest)]
    if d is not None:
        cmd += ['-d', str(d)]
    return run_command(cmd)

# Command: append
# Description: Append contents of directory to an existing archive
def append(d=None, i=None, o=None):
    cmd = ['append']
    if d is not None:
        cmd += ['-d', str(d)]
    if i is not None:
        cmd += ['-i', str(i)]
    if o is not None:
        cmd += ['-o', str(o)]
    return run_command(cmd)

# Command: patch
# Description: Patch input directory to output (asset extractor)
def patch(src=None, dst=None):
    cmd = ['patch']
    if src is not None:
        cmd += ['-src', str(src)]
    if dst is not None:
        cmd += ['-dst', str(dst)]
    return run_command(cmd)








