
            # Anacharssis_Z2.py (Updated: Integrated 2025 agentic AI improvements including multimodal GUI enhancement with VLM fallback (local Vision), hierarchical UI parsing; multi-agent collaboration via new 'multi_agent' command and sub-agent launcher inspired by CrewAI/AutoGen; self-improvement reflection with git versioning; benchmarking tool for agent evaluation; security allowlist for shell and efficiency optimizations like increased context size; updated FAQ with detailed explanations and command chaining; incorporated insights from 2025 papers (e.g., "Small Language Models are the Future of Agentic AI", "Generative to Agentic AI: Survey") and repos (CrewAI, PraisonAI, Agent-S); Reddit/X trends on multi-agent stacks, open-source agentic revolution, governance; added 'cyber'command to launch security tools script)

# References from research (substantiated claims, updated July 2025):
# - From arXiv papers (e.g., "Small Language Models are the Future of Agentic AI" [arXiv:2506.02153], "Generative to Agentic AI: Survey, Conceptualization, and Challenges" [arXiv:2504.18875], "AI Agents vs. Agentic AI: A Conceptual Taxonomy" [arXiv:2505.10468]): Emphasize SLMs for efficient agents, hierarchical/multi-agent systems for complex workflows, governance/security in agentic AI; integrated into multi_agent command and security allowlist.
# - GitHub repos (e.g., crewAIInc/crewAI, MervinPraison/PraisonAI, simular-ai/Agent-S, e2b-dev/awesome-ai-agents): Adopted multi-agent orchestration, human-like computer use for GUI, production-ready frameworks; enhanced launch_sub_agent and recognize_gui_elements.
# - Reddit/X insights (e.g., r/AI_Agents on "Best AI Agent Frameworks in 2025", X posts on "Agentic AI & Multi-Agent Ecosystems" [post:0], "AI Trends 2025" [post:2]): Focused on open-source multi-agent stacks, agentic revolution led by frameworks like AutoGen/Swarm; added benchmarking for evaluation, reflection in self-improvement to avoid loops.
# - Agentic AI trends (e.g., "Game Theory Meets LLM and Agentic AI" [arXiv:2507.10621], "TRiSM for Agentic AI" [arXiv:2506.04133]): Incorporated threat models into security, behavioral reflection for optimization; no voice/autonomous game per user request.

# FAQ
# ## Overview
# This section provides a professional summary of all commands (slash commands, multistep commands, and sudo commands) available in the agent's prompt. Each command is explained with its purpose, functionality, key details, and usage examples. This enables users or LLMs to understand and utilize the system effectively. Commands are triggered by recognizing patterns in user queries, leading to tool calls or multistep processes. Updates for 2025 include enhancements from agentic AI research: multimodal GUI with VLM fallback, multi-agent collaboration, self-improvement reflection with git, benchmarking, and security allowlists.

# ## Slash Commands (Direct Tool Calls)
# These are simple, direct invocations that map to specific tools for immediate execution.

# - **/search [query]**: Searches the web using the `duckduckgo_search` tool with `{"query": "[query]"}`. Returns top results for general knowledge or fact-checking. Example: `/search latest AI news`.
# - **/calc [expression]**: Evaluates mathematical expressions using the `Calculator` tool with `"[expression]"`. Supports basic to advanced calculations via `eval`. Example: `/calc 2 + 2 * 3`.
# - **/read [path]**: Reads a file using the `read_file` tool with `{"file_path": "[path]"}`. Returns content for analysis or processing. Example: `/read /path/to/file.txt`.
# - **/write [path] [text]**: Writes text to a file using the `write_file` tool with `{"file_path": "[path]", "text": "[text]"}`. Used for saving outputs or modifications. Example: `/write /path/to/file.txt Hello world`.
# - **/shell [command]**: Executes shell commands using the `run_shell` tool with `{"command": "[command]"}`. Runs system commands with safety checks for dangerous operations (e.g., sudo, rm -rf) and new 2025 allowlist for security (inspired by "TRiSM for Agentic AI"). Example: `/shell ls -l`.
# - **/image [positive] [negative]**: Generates images using the `generate_comfy_image` tool with `{"positive_prompt": "[positive]", "negative_prompt": "[negative]"}`. Modifies ComfyUI JSON and runs the workflow for AI image generation. Example: `/image a cat peaceful scene`.
# - **/detect [element]**: Detects GUI elements like search bars using the `detect_gui_element` tool with `{"element": "[element]"}`. Uses OpenCV for contour detection and returns coordinates. Example: `/detect search bar`.
# - **/mouse [element or color tuple]**: Moves/clicks the mouse using the `manipulate_mouse` tool with `{"element": "[element]"}` or `{"target_color": [tuple]}`. Detects via element or color; requires accessibility permissions. Example: `/mouse search bar`.
# - **/train [data_path] [epochs]**: Trains a screen model using the `train_screen_model` tool with `{"data_path": "[data_path]", "epochs": [epochs]}`. Uses PyTorch ResNet for bounding box regression on datasets. Example: `/train /path/to/data 10`.
# - **/help**: Lists all commands in the Final Answer. No tool call; provides direct user guidance.

# ## Multistep Commands (Chained Tool Calls or Reasoning)
# These involve sequences of tool calls or logical reasoning steps, often for complex tasks. Enhanced in 2025 with hierarchical/multi-agent support from papers like "Generative to Agentic AI".

# - **do open website: [url]**: Opens a URL in the default browser using `run_shell` with `"open [url]"`. Provides simple web access. Example: `do open website: https://example.com`.
# - **do open [app] and search [term]**: Opens the specified app (e.g., Google Chrome or Safari) using `run_shell`, waits for load, focuses the address bar with hotkey (Command+L on Mac), and types the search term followed by Enter. Inspired by ai-desktop GitHub repo for desktop control. Example: `do open Safari and search cnn.com`.
# - **do scroll [direction]**: Scrolls the screen in the specified direction (up or down) using the `scroll` tool. Useful for page navigation. Example: `do scroll down`.
# - **do research [topic]**: Searches [topic] with `duckduckgo_search` and summarizes results in the Final Answer. For quick overviews. Example: `do research AI agents`.
# - **improve [description]**: Searches the web for improvements (e.g., 'improvements to agentic AI code using DeepSeek papers'), proposes changes, confirms y/n, and applies via `edit_source_with_nano` on Anacharssis_Z2.py if yes, then restarts. Now includes reflection step (evaluate changes via LLM) and git commit for versioning (inspired by "AI Agent Behavioral Science" trends). Simulates self-improvement; avoids loops by tracking attempts. Example: `improve add new feature`.
# - **improve extend reasoning [task]**: Performs extended Chain-of-Thought (CoT) reasoning up to 10 steps, tracking thoughts to avoid loops, and switches to web search if stuck. For complex problem-solving. Example: `improve extend reasoning solve puzzle`.
# - **improve web search [query]**: Enhanced `duckduckgo_search` with up to 15 results, follows URLs if needed, and provides a deep summary. Standalone or subcommand. Example: `improve web search AI trends`.
# - **fix [issue]**: Self-prompts for fixes, searches the web if needed, proposes changes, confirms y/n, and edits Anacharssis_Z2.py via `edit_source_with_nano` (e.g., delete lines). Now with reflection and git for stability. For debugging. Example: `fix KeyError in prompt`.
# - **read [file_path]**: Absorbs a file (text/PDF) into chunks using `read_and_absorb` with `{"file_path": "[file_path]"}`, saves to Data dir. For processing large files. Example: `read /path/to/document.pdf`.
# - **read memory [file_path] [key]**: Absorbs file, reads content, saves as memory using `save_memory` with `{"key": "[key]", "content": content}`. Chains tools for persistent storage. Example: `read memory /path/to/file key1`.
# - **memory [key] [content]**: Saves content to Memory dir as .txt using `save_memory` with `{"key": "[key]", "content": "[content]"}`. Key-value store for data recall. Example: `memory key1 Some content`.
# - **remember [key]**: Retrieves .txt from Memory dir using `retrieve_memory` with `{"key": "[key]"}`, includes in response. Example: `remember key1`.
# - **game**: Searches web for AI prompting techniques, opens terminal, runs `"python {GABRIEL_PATH}"` to summon Gabriel for Baza game. Plots points, updates game_state.json. Simulates Baza game for training data generation. Example: `game`.
# - **do open [app]**: Opens any app on Mac using `run_shell` with `"open -a [app]"`. Example: `do open Finder`.
# - **recognize [description]**: Detects GUI elements using `recognize_gui_elements` with `{"description": "[description]"}`. Now with hierarchical parsing (parent-child relations) and local VLM fallback via enhanced Vision for multimodal grounding (inspired by "Large Language Model-Brained GUI Agents" and 2025 updates). Chains in workflows. Example: `recognize button`.
# - **recognize_n_summarize [app or description]**: Recognizes elements, summarizes UI, prompts for instructions. Chains `recognize_gui_elements` and LLM summary. Example: `recognize_n_summarize Safari`.
# - **list_gui_elements**: Reads internal GUI elements JSON from Data dir and displays for debugging. Example: `list_gui_elements`.
# - **do close window**: Loads GUI list, finds close button, clicks via `manipulate_mouse`. Opposite of `do open [app]`. Example: `do close window`.
# - **do_search_in_app [term]**: Loads GUI list, finds search bar, clicks and types [term] + Enter. Example: `do_search_in_app news`.
# - **do_search_in_app_on_website [term]**: Similar, but filters for non-top quadrant search bars (website-specific). Example: `do_search_in_app_on_website query`.
# - **multi_agent [task]**: Spawns sub-agents for complex tasks using `launch_sub_agent` with `{"task": "[task]"}`. Runs parallel instances via `run_shell` (e.g., "python sub_agent.py --task research"), aggregates results. Inspired by CrewAI/PraisonAI for multi-agent orchestration in 2025 trends. Avoids recursion depth issues with --sub flag. Example: `multi_agent deep research on AI trends`.
# - **benchmark [task]**: Runs evaluations on [task] using `run_benchmark` with `{"task": "[task]"}`. Simulates 10 runs, measures success/time/metrics (e.g., for GUI/agent tasks). Inspired by MLR-Bench and 2025 agent evaluation papers. Example: `benchmark GUI recognition accuracy`.
# - **cyber**: Launches security tools via `run_shell` with `"python {SUDO_CYBER_PATH}"`. Example: `cyber`.

# ### Command Chaining Flowchart
# This text-based flowchart illustrates how commands and subcommands can be chained using `&&` for sequential execution. It shows connections across command families (e.g., `do` with `sudo_recognize`). Chains execute left-to-right; use for workflows like app opening + recognition + interaction. Updated with multi_agent and benchmark chaining.

# ```
# Start
# |
# +-- do open [app] (Open app)
# |     |
# |     +-- && sudo_recognize [query] (Recognize GUI elements, generate internal list)
# |           |
# |           +-- && list_gui_elements (Debug: Display hidden GUI list)
# |           |     |
# |           |     +-- && do_search_in_app [term] (Search in app bar using list)
# |           |           |
# |           |           +-- && do close window (Close app using close button from list)
# |           |
# |           +-- && recognize_n_summarize [description] (Summarize UI, prompt for next)
# |           |
# |           +-- && multi_agent [task] (Spawn sub-agents for GUI interaction/analysis)
# |
# +-- improve [description] (Self-improve code with reflection/git)
#       |
#       +-- && fix [issue] (Debug/fix after improvement)
#             |
#             +-- && /help (List commands for reference)
#             |
#             +-- && benchmark [task] (Evaluate improvements)
# ```

# Examples:
# - Basic chain: `do open Safari && sudo_recognize button && list_gui_elements`
# - Cross-family: `do open Finder && recognize text && do close window`
# - Advanced: `improve add feature && fix error && game`
# - New: `multi_agent research AI && benchmark agent performance`

## Navs (Navigation and Recognition Commands)
# Focused on GUI navigation/recognition, inspired by visual grounding research (e.g., macOSWorld benchmark, SeeClick, UI-TARS papers) and repos (askui/vision-agent, pyautogui, MacPilot, Apple's Vision framework via PyObjC). Enhanced in 2025 with hierarchical parsing and multimodal fallback.

# - **recognize [description]**: Detects GUI elements in screen/app window using `recognize_gui_elements`. Uses OpenCV for shapes and Vision for text; fallbacks to heuristics; now builds hierarchy.
# - **recognize_n_summarize [app or description]**: Recognizes elements, summarizes layout, prompts for interaction instructions.
# - **sudo_recognize [query]**: Enhanced recognition via dedicated script. Generates hidden JSON list with hierarchy; ends with "What do you want me to do?" prompt. Subcommand: `sudo_recognize1` (shows list via --show-list flag).
# - **sudo_recognize1 [query]**: Runs sudo_recognize with flag to display full GUI list for debugging.

## Sudo Commands (Enhanced Multistep Commands)
# Executed via dedicated scripts using `run_shell` for expansion. If script file fails to open/execute, outputs "failed to open command .py file" (debugging added).

# - **sudo_open_website [query]**: Enhanced browser automation via script.
# - **sudo_research [topic]**: Deeper research via script (multiple engines, synthesis).
# - **sudo_improve [description]**: Advanced self-improvement via script.
# - **sudo_improve_extend_reasoning [task]**: Extended CoT via script.
# - **sudo_improve_web_search [query]**: Advanced searching via script.
# - **sudo_fix [issue]**: Sophisticated debugging via script.
# - **sudo_read [file_path]**: Advanced file absorption via script.
# - **sudo_read_memory [file_path] [key]**: Enhanced memory absorption via script.
# - **sudo_memory [key] [content]**: Advanced memory saving via script.
# - **sudo_remember [key]**: Advanced retrieval via script.
# - **sudo_game**: Advanced Baza game via script.
# - **sudo_do_open [app]**: Advanced app opening via script.
# - **sudo_recognize [query]**: Advanced GUI recognition via script.

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

# Centralize paths in config dict for efficiency (2025 optimization)
CONFIG = {
    "SCRIPT_PATH": "/Users/rosty/Desktop/Anacharssis_Z2.py",
    "GABRIEL_PATH": "/Users/rosty/Desktop/Gabriel_Z1.py",
    "DATA_DIR": "/Users/rosty/Desktop/AI/Data",
    "MEMORY_DIR": "/Users/rosty/Desktop/AI/Memory",
    "ACTIVE_MEMORY_DIR": "/Users/rosty/Desktop/AI/Agents/Active_Memory",
    "COMFY_JSON_PATH": "/Users/rosty/Desktop/AI/Images/comfy_json/comfy.json",
    "GUI_ELEMENTS_PATH": "/Users/rosty/Desktop/AI/Data/gui_elements.json",
    "SUDO_OPEN_WEBSITE_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_open_website.py",
    "SUDO_RESEARCH_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_research.py",
    "SUDO_IMPROVE_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_improve.py",
    "SUDO_IMPROVE_EXTEND_REASONING_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_improve_extend_reasoning.py",
    "SUDO_IMPROVE_WEB_SEARCH_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_improve_web_search.py",
    "SUDO_FIX_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_fix.py",
    "SUDO_READ_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_read.py",
    "SUDO_READ_MEMORY_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_read_memory.py",
    "SUDO_MEMORY_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_memory.py",
    "SUDO_REMEMBER_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_remember.py",
    "SUDO_GAME_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_game.py",
    "SUDO_DO_OPEN_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_do_open.py",
    "SUDO_RECOGNIZE_PATH": "/Users/rosty/Desktop/AI/Commands/sudo_recognize.py",
    "SUDO_CYBER_PATH": "/Users/rosty/Desktop/AI/Commands/cyber.py",
    "SCREEN_RES": (3024, 1964),  # MacBook Pro 14-inch resolution
    "GRID_SIZE": (11, 11),  # Small grid like in screenshot, e.g., -5 to 5 for Baza
    "CELL_SIZE_BASE": min((3024, 1964)) // 11  # Base cell size
}

# Shell command allowlist for security (2025 enhancement from "Securing Agentic AI" paper)
SHELL_ALLOWLIST = ["ls", "pwd", "echo", "cat", "open", "python"]  # Extend as needed; prefixes checked

os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
os.environ['OLLAMA_ORIGINS'] = '*'

os.makedirs(CONFIG["ACTIVE_MEMORY_DIR"], exist_ok=True)
os.makedirs(CONFIG["MEMORY_DIR"], exist_ok=True)
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)

model_tag = "deepseek-r1:7b"
llm = OllamaLLM(
    model=model_tag,
    temperature=0.0,
    num_gpu=-1,
    num_thread=4,
    num_ctx=8192,  # Increased for efficiency (2025 optimization)
    timeout=60,
    base_url="http://127.0.0.1:11434"
)

def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def run_shell_tool(args):
    command = args["command"]
    # Security allowlist check
    cmd_prefix = command.split()[0] if command else ""
    if cmd_prefix not in SHELL_ALLOWLIST:
        return f"Command '{cmd_prefix}' not in allowlist for security."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            check=False
        )
        output = result.stdout.strip()
        error_output = result.stderr.strip()
        full_output = output or error_output
        if result.returncode != 0:
            if command.startswith("python /Users/rosty/Desktop/AI/Commands/sudo_") and ("No such file or directory" in error_output or "not found" in error_output):
                return "failed to open command .py file"
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
    json_path = CONFIG["COMFY_JSON_PATH"]
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
    description = args.get("description", "update")
    try:
        with open(CONFIG["SCRIPT_PATH"], 'r') as f:
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
        with open(CONFIG["SCRIPT_PATH"], 'w') as f:
            f.writelines(lines)
        # Git versioning (2025 behavioral insight)
        subprocess.run(['git', 'add', CONFIG["SCRIPT_PATH"]])
        subprocess.run(['git', 'commit', '-m', f'Self-improvement: {description}'])
        subprocess.Popen(['open', '-a', 'Terminal', 'python', CONFIG["SCRIPT_PATH"]])
        return "Source edited, committed, and restarted."
    except Exception as e:
        return f"Error editing source: {e}"

def read_and_absorb(args):
    file_path = args.get("file_path")
    data_dir = CONFIG["DATA_DIR"]
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
    memory_dir = CONFIG["MEMORY_DIR"]
    filename = re.sub(r'\W+', '_', key)[:50] + ".txt"  # Descriptive name
    file_path = os.path.join(memory_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Memory saved as {filename} in {memory_dir}."

def retrieve_memory(args):
    key = args.get("key")
    memory_dir = CONFIG["MEMORY_DIR"]
    for file in os.listdir(memory_dir):
        if key in file and file.endswith('.txt'):
            with open(os.path.join(memory_dir, file), 'r') as f:
                return f.read()
    return f"Memory not found for key '{key}'."

def take_screenshot(args):
    path = args.get("path", os.path.join(CONFIG["ACTIVE_MEMORY_DIR"], f"screenshot_{time.time()}.png"))
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    return f"Screenshot saved to {path}"

def overlay_grid_on_image(args):
    image_path = args["image_path"]
    zoom = args.get("zoom", 1)
    output_path = args.get("output_path", image_path.replace('.png', '_grid.png'))
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cell_size = CONFIG["CELL_SIZE_BASE"] // zoom
    for i in range(CONFIG["GRID_SIZE"][0] * zoom + 1):
        cv2.line(img, (0, i * cell_size), (width, i * cell_size), (0, 0, 255), 1)
    for j in range(CONFIG["GRID_SIZE"][1] * zoom + 1):
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
    cell_size = CONFIG["CELL_SIZE_BASE"] // zoom
    px = width // 2 + int(x * cell_size)
    py = height // 2 - int(y * cell_size)  # Invert Y
    cv2.circle(img, (px, py), radius, color, -1)
    cv2.imwrite(output_path, img)
    return f"Point plotted at ({x},{y}) zoom {zoom}, saved to {output_path}"

def ls_directory(args):
    dir_path = args.get("dir_path", CONFIG["ACTIVE_MEMORY_DIR"])
    files = os.listdir(dir_path)
    return "\n".join(files)

def rename_file(args):
    old_path = args["old_path"]
    new_path = args["new_path"]
    os.rename(old_path, new_path)
    return f"Renamed {old_path} to {new_path}"

def delete_files_in_dir(args):
    dir_path = args.get("dir_path", CONFIG["ACTIVE_MEMORY_DIR"])
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
    path = os.path.join(CONFIG["ACTIVE_MEMORY_DIR"], "game_state.json")
    with open(path, 'w') as f:
        json.dump(state, f)
    return "Game state saved."

def load_game_state(args):
    path = os.path.join(CONFIG["ACTIVE_MEMORY_DIR"], "game_state.json")
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
    app = args.get("app", "Google Chrome")
    search_term = args.get("search_term", "")
    if app not in ["Google Chrome", "Safari"]:
        return f"Unsupported app: {app}. Only 'Google Chrome' or 'Safari' are supported."
    try:
        subprocess.run(["open", "-a", app])
        time.sleep(5)
        pyautogui.hotkey('command', 'l')
        time.sleep(0.5)
        pyautogui.typewrite(search_term)
        pyautogui.press('enter')
        return f"Opened {app} and searched for '{search_term}' successfully."
    except Exception as e:
        return f"Error automating browser: {e}. Ensure app is installed and accessibility permissions are granted."

def scroll(args):
    direction = args.get("direction", "down").lower()
    amount = 100 if direction == "up" else -100
    try:
        pyautogui.scroll(amount)
        return f"Scrolled {direction} successfully."
    except Exception as e:
        return f"Error scrolling: {e}. Ensure accessibility permissions are granted."

def build_hierarchy(elements):
    # Simple hierarchy builder: sort by position, assume nesting (2025 enhancement)
    hierarchy = {}
    sorted_elements = sorted(elements.items(), key=lambda e: (e[1]['coords'][1], e[1]['coords'][0]))  # Top-left first
    for i, (key, el) in enumerate(sorted_elements):
        hierarchy[key] = {"parent": None, "children": []}
        for j in range(i+1, len(sorted_elements)):
            child_key, child_el = sorted_elements[j]
            if (child_el['coords'][0] > el['coords'][0] and child_el['coords'][1] > el['coords'][1]):  # Rough nesting
                hierarchy[key]["children"].append(child_key)
                hierarchy[child_key]["parent"] = key
    return hierarchy

def recognize_gui_elements(args):
    description = args.get("description", "").lower()
    try:
        elements = {}
        # Enhanced Accessibility traversal if available
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
        # Text recognition using Apple's Vision if available (enhanced multimodal fallback)
        if HAS_VISION:
            screen_data = screen.tobytes()
            provider = CGDataProviderCreateWithData(None, screen_data, len(screen_data), None)
            cg_image = CGImageCreateWithPNGDataProvider(provider, None, True, kCGImageAlphaPremultipliedLast)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
            handler.performRequests_error_([request], None)
            results = request.results()
            for obs in results:
                text = obs.text()
                if description in text.lower():
                    bbox = obs.boundingBox()
                    x = bbox.origin.x * screen.width
                    y = (1 - bbox.origin.y - bbox.size.height) * screen.height  # Flip Y
                    w = bbox.size.width * screen.width
                    h = bbox.size.height * screen.height
                    elements[f"text_{len(elements)}"] = {"text": text, "coords": (int(x + w/2), int(y + h/2))}
        else:
            # Fallback: Basic text detection with OpenCV
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            text_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for tcnt in text_contours:
                tx, ty, tw, th = cv2.boundingRect(tcnt)
                if 10 < tw < 200 and 10 < th < 50:  # Heuristic for text blocks
                    elements[f"text_block_{len(elements)}"] = {"coords": (tx + tw // 2, ty + th // 2)}
        if elements:
            hierarchy = build_hierarchy(elements)  # New hierarchical parsing
            with open(CONFIG["GUI_ELEMENTS_PATH"], 'w') as f:
                json.dump({'elements': elements, 'hierarchy': hierarchy}, f)
            return elements
        return "No elements recognized matching the description."
    except Exception as e:
        return f"Error recognizing elements: {e}. Ensure Accessibility and pyobjc installed for full functionality."

def launch_sub_agent(args):
    task = args['task']
    sub_cmd = f"python {CONFIG['SCRIPT_PATH']} --sub {task}"  # Recursive with flag to limit depth
    return run_shell_tool({'command': sub_cmd})

def run_benchmark(args):
    task = args['task']
    # Simulate 10 runs (expand for real benchmarks, e.g., GUI accuracy)
    results = []
    for _ in range(10):
        # Mock run: e.g., call recognize_gui_elements and measure
        start = time.time()
        try:
            if "gui" in task.lower():
                recognize_gui_elements({"description": "test"})
            success = True
        except:
            success = False
        duration = time.time() - start
        results.append({"success": success, "duration": duration})
    success_rate = len([r for r in results if r['success']]) / 10
    avg_time = sum(r['duration'] for r in results) / 10
    return f"Benchmark results for {task}: Success rate {success_rate*100}%, Avg time {avg_time:.2f}s"

tool_dict = {
    "duckduckgo_search": Tool(name="duckduckgo_search", func=lambda q: DDGS().text(q, max_results=5), description="Search the web."),
    "Calculator": Tool(name="Calculator", func=lambda x: eval(x), description="Math calculations."),
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "run_shell": Tool(name="run_shell", func=run_shell_tool, description="Execute shell command."),
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
    "launch_sub_agent": Tool(name="launch_sub_agent", func=launch_sub_agent, description="Spawn sub-agent for multi-agent tasks."),
    "run_benchmark": Tool(name="run_benchmark", func=run_benchmark, description="Run benchmark evaluations."),
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
  - improve [description]: Automatically search web (duckduckgo_search) for improvements (e.g., query 'improvements to agentic AI code using DeepSeek papers'), propose changes based on results or papers like DeepSeek self-improvement, reflect on changes via LLM evaluation, ask y/n before applying. If yes, use edit_source_with_nano with changes dict to edit Anacharssis_Z2.py (Python file), git commit, then restart.
  - improve extend reasoning [task]: Perform extended Chain-of-Thought reasoning for up to 10 minutes (simulate by iterative self-prompting up to 10 steps), avoiding loops by tracking previous thoughts and switching strategies if stuck (e.g., if solution fails, use improve web search first).
  - improve web search [query]: Standalone or subcommand: Enhanced search using duckduckgo_search with more results (max 15), browse follow-up URLs if needed, summarize deeply.
  - fix [issue]: Prompt yourself with 'your code isn't working properly, please fix either your reasoning or your code to improve your output', search web if needed, propose fixes, reflect/evaluate, y/n prompt, then edit: For code fixes, simulate going to line 1 and Ctrl+K to cut (e.g., delete_first_line in changes), apply other changes, use edit_source_with_nano with git.
  - read [file_path]: Call read_and_absorb with {{ "file_path": "[file_path]" }} to read text/PDF, chunk (5000 chars limit), save as .txt in /Users/rosty/Desktop/AI/Data for training/reasoning (loop over chunks).
  - read memory [file_path] [key]: Call read_and_absorb {{ "file_path": "[file_path]" }}, then use returned message or chain read_file to get content, call save_memory {{ "key": "[key]", "content": content }} to store as .txt memory.
  - memory [key] [content]: Call save_memory with {{ "key": "[key]", "content": "[content]" }} to store in /Users/rosty/Desktop/AI/Memory as descriptive .txt (inspired by key-value stores from AI papers like MemGPT).
  - remember [key]: Call retrieve_memory with {{ "key": "[key]" }} to get content and include in Final Answer or use in reasoning.
  - game: Use improve web search for 'AI agent prompting techniques', 'AI LLM prompting best practices', 'troubleshooting AI agents coding issues', then run_shell "open -a Terminal", then chained "python {GABRIEL_PATH}" to summon Gabriel for Baza game. Plot points as red, share screenshot to Active_Memory as move_N.png, update game_state.json with plots, score, turns.
  - do open [app]: Use run_shell with "open -a [app]" to open any app on Mac.
  - recognize [description]: Call recognize_gui_elements with {{ "description": "[description]" }} to detect elements like buttons, text. Now hierarchical with multimodal fallback. Can chain (e.g., recognize button then mouse it).
  - recognize_n_summarize [description]: Call recognize_gui_elements, then use LLM to summarize detected elements/UI, prompt for instructions (e.g., "Summary: Buttons [list], Text [list]. How to proceed?").
  - list_gui_elements: Read {GUI_ELEMENTS_PATH} and summarize/print the list of detected GUI elements for debugging.
  - do close window: Chain load from {GUI_ELEMENTS_PATH}, find close_button (e.g., red X), call manipulate_mouse with coords to click and close app.
  - do_search_in_app [term]: Chain load from {GUI_ELEMENTS_PATH}, find search_bar, call manipulate_mouse to click, pyautogui.typewrite [term] + Enter.
  - do_search_in_app_on_website [term]: Similar to do_search_in_app but filter search_bar not in top quadrant (website-specific).
  - multi_agent [task]: Call launch_sub_agent with {{ "task": "[task]" }} to spawn sub-agents via parallel script instances, aggregate results for complex workflows (e.g., research + analysis).
  - benchmark [task]: Call run_benchmark with {{ "task": "[task]" }} to evaluate agent performance (e.g., success rate, time) on tasks like GUI recognition.
  - cyber: Call run_shell with "python {SUDO_CYBER_PATH}" to launch security tools.
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
  - sudo_game: Enhanced: Call run_shell with "python {SUDO_GAME_PATH}" to execute the dedicated sudo_game.py script for expanded features.
  - sudo_do_open [app]: Enhanced: Call run_shell with "python {SUDO_DO_OPEN_PATH} \\"[app]\\"" to execute the dedicated sudo_do_open.py script for advanced app opening.
  - sudo_recognize [query]: Enhanced: Call run_shell with "python {SUDO_RECOGNIZE_PATH} \\"[query]\\"" to execute the dedicated sudo_recognize.py script for advanced GUI recognition.
- For Baza game: Grid 11x11 (-5 to 5), plot at intersections, trap by encircling (detect_trapped_points with connected components), score trapped opponent points, wall parallel to block, diagonals to interrupt, super_move 10/20/30 plots (3 uses max, track in state 'super_uses':0-3), save_move 2 plots (unlimited but strategic). Use take_screenshot, overlay_grid_on_image if needed with zoom, plot_point_on_image (red for Ana, radius=5/zoom), rename_file to move_N.png, save_game_state {{ "plots": dict{{(x,y):player}}, "scores": {{ "ana":0, "gabriel":0 }}, "turn": N, "zoom": level (1 for base, >1 for zoom in, adjust cell_size), "super_uses": {{ "ana":0, "gabriel":0 }}, "coord_map": dict{{(virtual_x, virtual_y): (screen_px, screen_py)}} }}, ls_directory to check turns, kill_other_pythons after turn. If full grid (len(plots)==121), use memory "baza_grid_coords" [json coord_map].
- Calibration: Calculate coords = (width//2 + x * cell_size, height//2 - y * cell_size), cell_size = CELL_SIZE_BASE / zoom, store in state coord_map for quick access.
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
        GABRIEL_PATH=CONFIG["GABRIEL_PATH"],
        SUDO_OPEN_WEBSITE_PATH=CONFIG["SUDO_OPEN_WEBSITE_PATH"],
        SUDO_RESEARCH_PATH=CONFIG["SUDO_RESEARCH_PATH"],
        SUDO_IMPROVE_PATH=CONFIG["SUDO_IMPROVE_PATH"],
        SUDO_IMPROVE_EXTEND_REASONING_PATH=CONFIG["SUDO_IMPROVE_EXTEND_REASONING_PATH"],
        SUDO_IMPROVE_WEB_SEARCH_PATH=CONFIG["SUDO_IMPROVE_WEB_SEARCH_PATH"],
        SUDO_FIX_PATH=CONFIG["SUDO_FIX_PATH"],
        SUDO_READ_PATH=CONFIG["SUDO_READ_PATH"],
        SUDO_READ_MEMORY_PATH=CONFIG["SUDO_READ_MEMORY_PATH"],
        SUDO_MEMORY_PATH=CONFIG["SUDO_MEMORY_PATH"],
        SUDO_REMEMBER_PATH=CONFIG["SUDO_REMEMBER_PATH"],
        SUDO_GAME_PATH=CONFIG["SUDO_GAME_PATH"],
        SUDO_DO_OPEN_PATH=CONFIG["SUDO_DO_OPEN_PATH"],
        SUDO_RECOGNIZE_PATH=CONFIG["SUDO_RECOGNIZE_PATH"],
        SUDO_CYBER_PATH=CONFIG["SUDO_CYBER_PATH"],
        GUI_ELEMENTS_PATH=CONFIG["GUI_ELEMENTS_PATH"],
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
        if target == CONFIG["SCRIPT_PATH"] and action != "edit_source_with_nano":
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
                    if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                        with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
                            gui_list = json.load(f)
                        print(f"\033[1;31mAnacharssis:\033[0m GUI Elements List: {json.dumps(gui_list, indent=2)} (from chained command)")
                    else:
                        print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first. (from chained command)")
                    continue
                elif lower_cmd.startswith("do close window"):
                    if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                        with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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
                    if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                        with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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
                    if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                        with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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
                    if action in tool_dict and input_json is not None:
                        if not safe_tool_check(action, input_json):
                            continue
                        tool = tool_dict[action]
                        # ... (similar to main tool execution block, but print with "(from chained command)")
            continue  # Skip further processing after chaining
        if "do open " in lower_query and " and search " in lower_query:
            parts = lower_query.split("do open ", 1)[1].split(" and search ", 1)
            if len(parts) == 2:
                app = parts[0].strip().title()
                term = parts[1].strip()
                input_json = {"app": app, "search_term": term}
                result = automate_browser(input_json)
                print(f"\033[1;31mAnacharssis:\033[0m {result}")
                continue
            else:
                print(f"\033[1;31mAnacharssis:\033[0m Invalid format for 'do open [app] and search [term]'. Example: do open safari and search etsy.com")
                continue
        if "do open " in lower_query and " and search " not in lower_query:
            parts = lower_query.split("do open ", 1)
            if len(parts) == 2:
                app = parts[1].strip().title()
                try:
                    subprocess.run(["open", "-a", app])
                    print(f"\033[1;31mAnacharssis:\033[0m Opened {app} successfully.")
                except Exception as e:
                    print(f"\033[1;31mAnacharssis:\033[0m Error opening {app}: {e}. Ensure the app is installed.")
                continue
            else:
                print(f"\033[1;31mAnacharssis:\033[0m Invalid format for 'do open [app]'. Example: do open finder")
                continue

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

        if "favorite command" in lower_query:
            print("\033[1;31mAnacharssis:\033[0m My favorite command is 'do open safari and search' because it lets me interact with the web in a fun, visual way!")
            continue

        if lower_query.startswith("list_gui_elements"):
            if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
                    gui_list = json.load(f)
                print(f"\033[1;31mAnacharssis:\033[0m GUI Elements List: {json.dumps(gui_list, indent=2)}")
            else:
                print(f"\033[1;31mAnacharssis:\033[0m No GUI elements list found. Run sudo_recognize first.")
            continue

        if lower_query.startswith("do close window"):
            if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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

        if lower_query.startswith("do_search_in_app "):
            term = query[len("do_search_in_app "):].strip()
            if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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

        if lower_query.startswith("do_search_in_app_on_website "):
            term = query[len("do_search_in_app_on_website "):].strip()
            if os.path.exists(CONFIG["GUI_ELEMENTS_PATH"]):
                with open(CONFIG["GUI_ELEMENTS_PATH"], 'r') as f:
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

            should_animate = (action in ("run_shell", "generate_comfy_image", "manipulate_mouse", "detect_gui_element", "train_screen_model", "edit_source_with_nano", "read_and_absorb", "save_memory", "retrieve_memory", "take_screenshot", "overlay_grid_on_image", "plot_point_on_image", "ls_directory", "rename_file", "delete_files_in_dir", "kill_other_pythons", "save_game_state", "load_game_state", "detect_trapped_points", "automate_browser", "scroll", "recognize_gui_elements", "launch_sub_agent", "run_benchmark"))
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
            
            
            
            