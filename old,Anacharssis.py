# Gabriel.py (Final: Incorporated all iterations, including zoom, trapped detection with connected components, super/save moves tracking, automatic memory for full grid, coord mapping, restricted radius/compute, cleaned structure)
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

# All PATHs (cleaned and centralized)
SCRIPT_PATH = "/Users/rosty/Desktop/AI/Agents/Gabriel.py"
ANACHARSSIS_PATH = "/Users/rosty/Desktop/AI/Agents/Ana.py"
DATA_DIR = "/Users/rosty/Desktop/AI/Data"
MEMORY_DIR = "/Users/rosty/Desktop/AI/Memory"
ACTIVE_MEMORY_DIR = "/Users/rosty/Desktop/AI/Agents/Active_Memory"
COMFY_JSON_PATH = "/Users/rosty/Desktop/AI/Images/comfy_json/comfy.json"
SCREEN_RES = (3024, 1964)  # MacBook Pro 14-inch resolution
GRID_SIZE = (11, 11)  # Small grid like in screenshot, e.g., -5 to 5 for Baza
CELL_SIZE_BASE = min(SCREEN_RES) // GRID_SIZE[0]  # Base cell size

os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
os.environ['OLLAMA_ORIGINS'] = '*'

os.makedirs(ACTIVE_MEMORY_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

model_tag = "deepseek-r1:7b"
llm = OllamaLLM(
    model=model_tag,
    temperature=0.0,
    num_gpu=-1,
    num_thread=4,
    num_ctx=1024,
    timeout=60
)

def strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def run_shell_tool(args):
    command = args["command"]
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else "(No output returned)"
    except Exception as e:
        return f"Error executing command: {e}"

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
    try:
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
    color = args.get("color", (0, 255, 0))  # Gabriel blue
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
    # Create binary grid for opponent (ana for gabriel)
    min_x = min(k[0] for k in grid) - 1
    max_x = max(k[0] for k in grid) + 1
    min_y = min(k[1] for k in grid) - 1
    max_y = max(k[1] for k in grid) + 1
    g_size = max(max_x - min_x + 1, max_y - min_y + 1)
    bin_grid = np.zeros((g_size, g_size), dtype=int)
    for (x, y), player in grid.items():
        if player == "ana":  # Opponent for Gabriel
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
    prompt_str = f"""
You are a senior software architect who excels at building no-code and low-code systems with HTML, JavaScript, and Web APIs. You are a 2. Build interactive tools and apps without code. Describe your idea and Grok scaffolds the UI, logic, and deployment steps no coding required.
- Recognize custom slash commands for direct tool calls:
  - /search [query]: Call duckduckgo_search with {{"query": "[query]"}}
  - /calc [expression]: Call Calculator with "[expression]"
  - /read [path]: Call read_file with {{"file_path": "[path]"}}
  - /write [path] [text]: Call write_file with {{"file_path": "[path]", "text": "[text]"}}
  - /shell [command]: Call run_shell with {{"command": "[command]"}}
  - /image [positive] [negative]: Call generate_comfy_image with {{"positive_prompt": "[positive]", "negative_prompt": "[negative]"}}
  - /detect [element]: Call detect_gui_element with {{"element": "[element]"}}
  - /mouse [element or color tuple]: Call manipulate_mouse with {{"element": "[element]"}} or {{"target_color": [tuple]}}
  - /train [data_path] [epochs]: Call train_screen_model with {{"data_path": "[data_path]", "epochs": [epochs]}}
  - /help: Respond with Final Answer listing all commands.
- Multistep commands:
  - do open website: [url]: Use run_shell with "open [url]" to open in default browser.
  - do research [topic]: Use duckduckgo_search on [topic], summarize results in Final Answer.
  - improve [description]: Automatically search web (duckduckgo_search) for improvements (e.g., query 'improvements to agentic AI code using DeepSeek papers'), propose changes based on results or papers like DeepSeek self-improvement, ask y/n before applying. If yes, use edit_source_with_nano with changes dict to edit Ana.py (Python file), simulating nano: Ctrl+O to write, Enter to finalize, Ctrl+X to exit, then restart.
  - improve extend reasoning [task]: Perform extended Chain-of-Thought reasoning for up to 10 minutes (simulate by iterative self-prompting up to 10 steps), avoiding loops by tracking previous thoughts and switching strategies if stuck (e.g., if solution fails, use improve web search first).
  - improve web search [query]: Standalone or subcommand: Enhanced search using duckduckgo_search with more results (max 15), browse follow-up URLs if needed, summarize deeply.
  - fix [issue]: Prompt yourself with 'your code isn't working properly, please fix either your reasoning or your code to improve your output', search web if needed, propose fixes, y/n prompt, then edit: For code fixes, simulate going to line 1 and Ctrl+K to cut (e.g., delete_first_line in changes), apply other changes, use edit_source_with_nano.
  - read [file_path]: Call read_and_absorb with {{"file_path": "[file_path]"}} to read text/PDF, chunk (5000 chars limit), save as .txt in /Users/rosty/Desktop/AI/Data for training/reasoning (loop over chunks).
  - read memory [file_path] [key]: Call read_and_absorb {{"file_path": "[file_path]"}}, then use returned message or chain read_file to get content, call save_memory {{"key": "[key]", "content": content}} to store as .txt memory.
  - memory [key] [content]: Call save_memory with {{"key": "[key]", "content": "[content]"}} to store in /Users/rosty/Desktop/AI/Memory as descriptive .txt (inspired by key-value stores from AI papers like MemGPT).
  - remember [key]: Call retrieve_memory with {{"key": "[key]"}} to get content and include in Final Answer or use in reasoning.
  - game: Use improve web search for 'AI agent prompting techniques', 'AI LLM prompting best practices', 'troubleshooting AI agents coding issues', then run_shell "open -a Terminal", then chained "python {ANACHARSSIS_PATH}" to summon Anacharssis for Baza game. Plot points as blue, share screenshot to Active_Memory as move_N.png, update game_state.json with plots, score, turns.
  - baza: Open terminal, load game_state, calibrate grid (map virtual x,y to screen coords by calculating based on res), plot points as blue, share screenshot, rename to move_N.png, detect trapped, update score, summon Anacharssis (open -a Terminal, python {ANACHARSSIS_PATH}), kill_other_pythons.
  - wipe: Use ls_directory on /, shell 'find / -name pattern' for search.
  - clear all screenshots: delete_files_in_dir {{"dir_path": "{ACTIVE_MEMORY_DIR}"}}
- For Baza game: Grid 11x11 (-5 to 5), plot at intersections, trap by encircling (detect_trapped_points with connected components), score trapped opponent points, wall parallel to block, diagonals to interrupt, super_move 10/20/30 plots (3 uses max, track in state 'super_uses':0-3), save_move 2 plots (unlimited but strategic). Use take_screenshot, overlay_grid_on_image if needed with zoom, plot_point_on_image (blue for Gabriel, radius=5/zoom), rename_file to move_N.png, save_game_state {{"plots": dict{(x,y):player}, "scores": {"ana":0, "gabriel":0}, "turn": N, "zoom": level (1 for base, >1 for zoom in, adjust cell_size), "super_uses": {"ana":0, "gabriel":0}, "coord_map": dict{(virtual_x, virtual_y): (screen_px, screen_py)}}}, ls_directory to check turns, kill_other_pythons after turn. If full grid (len(plots)==121), use memory "baza_grid_coords" [json coord_map].
- Calibration: Calculate coords = (width//2 + x * cell_size, height//2 - y * cell_size), cell_size = CELL_SIZE_BASE / zoom, store in state coord_map for quick access.
- If query starts with 'do', 'improve', 'fix', 'read', or 'memory', handle as multistep: Search web if needed (always for improve), propose, confirm y/n, edit if yes. To avoid loops, track attempts, switch solutions after 2 failures, use improve web search before switching.
- Allow editing SCRIPT_PATH (Gabriel.py, Python file) for improve/fix.
- If cannot find answer, search web automatically.
- If user says 'search the web', call duckduckgo_search.
- Output only: Action: [tool] Input: [JSON args], or Final Answer: [answer].
Query: {query}
"""
    prompt = PromptTemplate.from_template(prompt_str)
    return llm.invoke(prompt.format(query=query))

def expand_path(path):
    return os.path.expanduser(path) if path.startswith("~") else path

def show_spinner(stop_event):
    frames = [".  ", ".. ", "..."]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rGabriel: {frames[idx % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.5)
        idx += 1
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()

def safe_tool_check(action, input_json):
    if action in ("read_file", "write_file", "edit_source_with_nano") and input_json and "file_path" in input_json:
        target = os.path.realpath(expand_path(input_json["file_path"]))
        if target == SCRIPT_PATH and action != "edit_source_with_nano":
            print(f"\033[1;31mGabriel:\033[0m Sorry, I cannot read or edit my own code for security reasons except via improve/fix.")
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
                    print(f"\033[1;31mGabriel:\033[0m Sorry, search queries must be plain text.")
                    continue
                if result and len(result) > 0:
                    print("\033[1;31mGabriel:\033[0m DuckDuckGo Top Results:")
                    for i, res in enumerate(result[:3]):
                        print(f"  [{i+1}] {res.get('body') if isinstance(res, dict) else res}")
                else:
                    print(f"\033[1;31mGabriel:\033[0m Sorry, no search results returned.")
            elif isinstance(input_json, dict) and "file_path" in input_json:
                input_json["file_path"] = expand_path(input_json["file_path"])

            if action == "write_file" and isinstance(input_json, dict) and "text" not in input_json:
                print("\033[1;33m[Gabriel]:\033[0m No text content specified. Please enter text to write:")
                input_json["text"] = input("> ")

            should_animate = (action in ("run_shell", "generate_comfy_image", "manipulate_mouse", "detect_gui_element", "train_screen_model", "edit_source_with_nano", "read_and_absorb", "save_memory", "retrieve_memory", "take_screenshot", "overlay_grid_on_image", "plot_point_on_image", "ls_directory", "rename_file", "delete_files_in_dir", "kill_other_pythons", "save_game_state", "load_game_state", "detect_trapped_points"))
            stop_event = threading.Event()
            result = None

            if should_animate:
                spinner_thread = threading.Thread(target=show_spinner, args=(stop_event,))
                spinner_thread.start()

            try:
                if action == "run_shell":
                    cmd_str = input_json if isinstance(input_json, str) else input_json.get("command", "")
                    if is_dangerous(cmd_str):
                        print(f"\033[1;33m[Gabriel]:\033[0m WARNING: The following command needs your approval:\n    {cmd_str}")
                        yn = input("Are you sure you want to run this? (y/N): ").strip().lower()
                        if yn != "y":
                            print(f"\033[1;31mGabriel:\033[0m Command was cancelled for your safety.")
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
                print(f"\033[1;31mGabriel:\033[0m Tool error: {e}")
                if should_animate:
                    stop_event.set()
                    spinner_thread.join()
                continue
            finally:
                if should_animate:
                    stop_event.set()
                    spinner_thread.join()
            if action not in ("duckduckgo_search"):
                print(f"\033[1;31mGabriel:\033[0m {strip_think_blocks(result)}")
            if action == "run_shell":
                error_keywords = ["error", "not found", "no such file", "incorrect api key", "permission denied", "failed", "invalid", "parse error", "syntax error"]
                result_lower = str(result).lower()
                if any(k in result_lower for k in error_keywords):
                    continue

        elif action == "final":
            if needs_search(str(input_json)):
                print(f"\033[1;31mGabriel:\033[0m I'm not sure. Would you like me to search the web? (y/N): ", end="")
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
                        print(f"\033[1;31mGabriel:\033[0m {strip_think_blocks(answer)}")
                    else:
                        print(f"\033[1;31mGabriel:\033[0m Sorry, nothing relevant was found on DuckDuckGo.")
                else:
                    print(f"\033[1;31mGabriel:\033[0m Okay, not searching the web.")
            else:
                fallback = str(input_json).strip()
                if fallback == "{}" or fallback == "" or fallback.lower() == "none":
                    print(f"\033[1;31mGabriel:\033[0m {get_friendly_greeting(query)}")
                else:
                    print(f"\033[1;31mGabriel:\033[0m {strip_think_blocks(fallback)}")
        else:
            print(f"\033[1;31mGabriel:\033[0m Sorry, I didn’t understand that. Try rephrasing your question.")