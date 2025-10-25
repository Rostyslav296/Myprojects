#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniNano (Windows-only) — unified console, robust line numbers, full dark mode, inline Ctrl+F

- Windows only (all *nix code removed)
- One PowerShell console per tab (no separate GPT tab)
- Two input bars per console tab:
    • Command -> executes in PowerShell
    • Prompt  -> prints text to console (no exec)
- Browser-style "+" tab for new consoles; close tab by clicking the left ~16px,
  middle-click, or Ctrl+W
- Canvas-based line-number gutter (no flicker or missing lines)
- Dark theme applied to editor, tree, tabs, buttons, entries, scrollbars, menus, status bar
- Inline Find bar (Ctrl+F): next/prev, count; Enter=next, Shift+Enter=prev, F3/Shift+F3 too
- All PowerShell/background processes run hidden (no stray cmd windows)
"""
import os, sys, threading, subprocess, json, shutil, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont

if os.name != "nt":
    raise SystemExit("This build of MiniNano is Windows-only.")

APP_TITLE = "MiniNano"
MONO_FONT = ("Consolas", 11)

# ---------- Config ----------
CFG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "mininano"
CFG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULTS = {
    "dark_mode": True,
    "pins": [None, None, None, None, None],
    "fb_root": None
}
CFG_PATH = CFG_DIR / "config.json"

def load_cfg():
    if CFG_PATH.exists():
        try:
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return DEFAULTS.copy()

def save_cfg(cfg: dict):
    try:
        CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------- Themes ----------
LIGHT = {
    "bg": "#ffffff", "fg": "#1e1e1e", "sel_bg": "#cde8ff", "sel_fg": "#000000",
    "cursor": "#222222", "console_bg": "#fafafa", "console_fg": "#111111",
    "toolbar_bg": "#e9e9e9", "status_bg": "#efefef", "status_fg": "#333333",
    "search_bg": "#fff7a8", "search_fg": "#000000",
    "gutter_bg": "#f3f3f3", "gutter_fg": "#666666",
    "menu_bg": "#ffffff", "menu_fg": "#1e1e1e", "menu_active": "#e9e9e9",
    "tab_bg": "#e3e3e3", "tab_fg": "#1e1e1e", "tab_sel_bg": "#d7d7d7", "tab_sel_fg": "#000000",
    "tree_sel_bg": "#cde8ff", "tree_sel_fg": "#000000",
}
DARK = {
    "bg": "#1e1e1e", "fg": "#d4d4d4", "sel_bg": "#264f78", "sel_fg": "#ffffff",
    "cursor": "#cccccc", "console_bg": "#111317", "console_fg": "#d6dde6",
    "toolbar_bg": "#2b2d31", "status_bg": "#2b2d31", "status_fg": "#c7c7c7",
    "search_bg": "#3a3d41", "search_fg": "#ffd866",
    "gutter_bg": "#252526", "gutter_fg": "#8e8e8e",
    "menu_bg": "#1e1e1e", "menu_fg": "#d4d4d4", "menu_active": "#2b2d31",
    "tab_bg": "#2b2d31", "tab_fg": "#cfd3da", "tab_sel_bg": "#3a3d41", "tab_sel_fg": "#ffffff",
    "tree_sel_bg": "#264f78", "tree_sel_fg": "#ffffff",
}

CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

class MiniNano(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(900, 560)

        # State
        self.cfg = load_cfg()
        self.theme = DARK if self.cfg.get("dark_mode", True) else LIGHT
        self.filename: Path | None = None
        self._modified = False
        self.project_root: Path | None = None
        self.fb_root: Path | None = None
        self.fs_clipboard = {"mode": None, "paths": []}
        self.consoles = []  # [{frame, text, cmd_entry, prompt_entry, proc}]
        self._ln_font = tkfont.Font(master=self, font=MONO_FONT)  # gutter font (explicit master)
        # inline find state
        self.find_term = ""
        self.find_matches = []
        self.find_index = -1

        # UI
        self._build_ui()
        self._apply_theme()
        self._bind_keys()
        self.new_file()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        # Menu
        menubar = tk.Menu(self)
        file_m = tk.Menu(menubar, tearoff=False)
        file_m.add_command(label="New", accelerator="Ctrl+N", command=self.new_file)
        file_m.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_file)
        file_m.add_separator()
        file_m.add_command(label="Save", accelerator="Ctrl+S", command=self.save_file)
        file_m.add_command(label="Save As...", command=self.save_file_as)
        file_m.add_separator()
        file_m.add_command(label="Run (F5)", accelerator="F5", command=self.run_current)
        file_m.add_command(label="Stop", accelerator="Esc", command=self.stop_run)
        file_m.add_separator()
        file_m.add_command(label="Restart", command=self.restart_app)
        file_m.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_m)

        project_m = tk.Menu(menubar, tearoff=False)
        project_m.add_command(label="Set Project Root...", command=self._set_project_root)
        project_m.add_command(label="Set File Browser Root...", command=self._set_fb_root)
        project_m.add_command(label="Refresh File Browser", command=self._fb_refresh)
        menubar.add_cascade(label="Project", menu=project_m)

        view_m = tk.Menu(menubar, tearoff=False)
        self.dark_mode_var = tk.BooleanVar(value=self.cfg.get("dark_mode", True))
        view_m.add_checkbutton(label="Dark Mode", onvalue=True, offvalue=False,
                               variable=self.dark_mode_var, command=self.toggle_dark_mode,
                               accelerator="Ctrl+Shift+D")
        menubar.add_cascade(label="View", menu=view_m)

        help_m = tk.Menu(menubar, tearoff=False)
        help_m.add_command(label="About", command=lambda: messagebox.showinfo(
            "About MiniNano",
            "MiniNano — Windows PowerShell console with browser-style tabs, two input bars (Command & Prompt)."
        ))
        menubar.add_cascade(label="Help", menu=help_m)
        self.config(menu=menubar)

        # Toolbar
        self.toolbar_outer = tk.Frame(self, bd=0, highlightthickness=0)
        self.toolbar_outer.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = ttk.Frame(self.toolbar_outer, padding=(6, 6))
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(self.toolbar, text="Open", command=self.open_file)
        self.btn_save = ttk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_file)
        self.btn_save_as = ttk.Button(self.toolbar, text="Save As...", command=self.save_file_as)
        self.btn_undo = ttk.Button(self.toolbar, text="Undo (Ctrl+Z)", command=lambda: self.text.event_generate("<<Undo>>"))
        self.btn_redo = ttk.Button(self.toolbar, text="Redo (Ctrl+Y)", command=lambda: self.text.event_generate("<<Redo>>"))
        self.btn_run = ttk.Button(self.toolbar, text="Run (F5)", command=self.run_current)
        self.btn_stop = ttk.Button(self.toolbar, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.btn_copyall = ttk.Button(self.toolbar, text="Copy All", command=self.copy_all)
        self.btn_restart = ttk.Button(self.toolbar, text="Restart", command=self.restart_app)
        for w in (self.btn_open, self.btn_save, self.btn_save_as, self.btn_undo, self.btn_redo,
                  self.btn_run, self.btn_stop, self.btn_copyall, self.btn_restart):
            w.pack(side=tk.LEFT, padx=(0, 6))

        # Pin buttons
        self.pin_buttons_frame = ttk.Frame(self.toolbar)
        self.pin_buttons_frame.pack(side=tk.RIGHT)
        self.pin_buttons = []
        for i in range(5):
            btn = ttk.Button(self.pin_buttons_frame, text=f"Pin{i+1}", width=6)
            btn.bind("<Button-1>", lambda e, i=i: self._pin_button_click(i, e))
            btn.pack(side=tk.LEFT, padx=2)
            self.pin_buttons.append(btn)
        self._update_pin_button_labels()

        # Main layout
        self.root_paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.root_paned.pack(fill=tk.BOTH, expand=True)

        # Top: File Browser | Editor
        self.top_paned = ttk.PanedWindow(self.root_paned, orient=tk.HORIZONTAL)
        self.root_paned.add(self.top_paned, weight=5)

        # --- File Browser
        self.fb_panel = ttk.Frame(self.top_paned, width=260)
        self._build_file_browser(self.fb_panel)
        self.top_paned.add(self.fb_panel, weight=1)

        # --- Editor + gutter (canvas-based line numbers)
        editor_outer = ttk.Frame(self.top_paned)
        self.top_paned.add(editor_outer, weight=5)

        self.gutter_canvas = tk.Canvas(editor_outer, width=40, highlightthickness=0, bd=0)
        self.gutter_canvas.pack(side=tk.LEFT, fill=tk.Y)

        editor_frame = ttk.Frame(editor_outer)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.text = tk.Text(editor_frame, wrap="none", undo=True, font=MONO_FONT, tabs=("1c",), insertwidth=2)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(editor_frame, orient="vertical", command=self._yview_text)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=lambda *a: self._yscrollcommand(yscroll, *a))

        # Search highlight tag
        self.text.tag_configure("search_hit")

        # --- Inline Find Bar (hidden by default; inline, not a popup)
        self.find_bar = ttk.Frame(editor_outer, padding=(6, 4))
        self.find_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.find_bar.pack_forget()

        ttk.Label(self.find_bar, text="Find:").pack(side=tk.LEFT, padx=(0, 6))
        self.find_entry = ttk.Entry(self.find_bar, width=32)
        self.find_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.find_prev_btn = ttk.Button(self.find_bar, text="◄ Prev", command=self._find_prev)
        self.find_prev_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.find_next_btn = ttk.Button(self.find_bar, text="Next ►", command=self._find_next)
        self.find_next_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.find_count = ttk.Label(self.find_bar, text="0/0")
        self.find_count.pack(side=tk.LEFT)

        self.find_entry.bind("<Return>", lambda e: (self._find_next(), "break"))
        self.find_entry.bind("<Shift-Return>", lambda e: (self._find_prev(), "break"))

        # Bottom: Console Notebook with "+" tab
        self.console_tabs = ttk.Notebook(self.root_paned)
        self.root_paned.add(self.console_tabs, weight=2)
        self._add_plus_tab()
        self._install_tab_close_behavior()
        self.new_console_tab()  # Start with one console

        # Status bar
        self.status_outer = tk.Frame(self, bd=0, highlightthickness=0)
        self.status_outer.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(self.status_outer, anchor="w", padding=(6, 3))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Bindings
        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<KeyRelease>", lambda e: (self._update_title_status(), self._schedule_line_numbers()))
        self.text.bind("<ButtonRelease-1>", lambda e: (self._update_title_status(), self._schedule_line_numbers()))
        self.text.bind("<Configure>", lambda e: self._schedule_line_numbers())
        self.text.bind("<MouseWheel>", lambda e: self._schedule_line_numbers())

        self.console_tabs.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self.console_tabs.bind("<ButtonRelease-1>", self._maybe_create_from_plus)

    # ----- Inline Find (Ctrl+F) -----
    def _toggle_find(self):
        if self.find_bar.winfo_ismapped():
            self._close_find()
        else:
            self.find_bar.pack(side=tk.BOTTOM, fill=tk.X)
            # seed with current selection if any
            try:
                sel = self.text.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                sel = ""
            self.find_entry.delete(0, tk.END)
            if sel:
                self.find_entry.insert(0, sel)
            self.find_entry.focus_set()
            self._apply_theme()  # ensure colors match

    def _close_find(self):
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.find_bar.pack_forget()
        self.find_term = ""
        self.find_matches = []
        self.find_index = -1
        self.find_count.config(text="0/0")
        self.text.focus_set()

    def _find_all(self, term: str):
        self.find_matches = []
        if not term:
            return
        start = "1.0"
        while True:
            pos = self.text.search(term, start, stopindex=tk.END, nocase=1)
            if not pos:
                break
            end_pos = f"{pos}+{len(term)}c"
            self.find_matches.append((pos, end_pos))
            start = end_pos

    def _find_next(self):
        term = self.find_entry.get().strip()
        if not term:
            return
        if term != self.find_term:
            self.find_term = term
            self._find_all(term)
            self.find_index = -1
        if not self.find_matches:
            self.find_count.config(text="0/0"); return
        self.find_index = (self.find_index + 1) % len(self.find_matches)
        pos, endp = self.find_matches[self.find_index]
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.text.tag_add("search_hit", pos, endp)
        self.text.see(pos)
        self.find_count.config(text=f"{self.find_index + 1}/{len(self.find_matches)}")

    def _find_prev(self):
        term = self.find_entry.get().strip()
        if not term:
            return
        if term != self.find_term:
            self.find_term = term
            self._find_all(term)
            self.find_index = -1
        if not self.find_matches:
            self.find_count.config(text="0/0"); return
        if self.find_index == -1:
            self.find_index = len(self.find_matches) - 1
        else:
            self.find_index = (self.find_index - 1) % len(self.find_matches)
        pos, endp = self.find_matches[self.find_index]
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.text.tag_add("search_hit", pos, endp)
        self.text.see(pos)
        self.find_count.config(text=f"{self.find_index + 1}/{len(self.find_matches)}")

    # ---------- Console tabs "+" and close behavior ----------
    def _add_plus_tab(self):
        plus = ttk.Frame(self.console_tabs)
        self.console_tabs.add(plus, text="+")
        self.plus_tab = plus

    def _install_tab_close_behavior(self):
        # Click in left ~16px of a tab to close; middle-click closes; Ctrl+W closes active
        self.console_tabs.bind("<Button-1>", self._on_notebook_click, add="+")
        self.console_tabs.bind("<Button-2>", self._on_notebook_middle_click, add="+")
        self.console_tabs.bind("<Button-3>", self._on_notebook_middle_click, add="+")
        self.bind_all("<Control-w>", lambda e: (self._close_active_tab(), "break"))

        # Find shortcuts (inline)
        self.bind_all("<Control-f>", lambda e: (self._toggle_find(), "break"))
        self.bind_all("<F3>", lambda e: (self._find_next(), "break"))
        self.bind_all("<Shift-F3>", lambda e: (self._find_prev(), "break"))
        self.bind_all("<Escape>", self._handle_escape, add="+")

    def _handle_escape(self, event):
        if self.find_bar.winfo_ismapped():
            self._close_find()
            return "break"
        self.stop_run()
        return "break"

    def _on_notebook_click(self, event):
        try:
            idx = self.console_tabs.index(f"@{event.x},{event.y}")
        except Exception:
            return
        tab_id = self.console_tabs.tabs()[idx]
        if tab_id == str(self.plus_tab):
            return
        x, y, w, h = self.console_tabs.bbox(idx)
        if event.x <= x + 16:
            self._close_tab_by_index(idx)

    def _on_notebook_middle_click(self, event):
        try:
            idx = self.console_tabs.index(f"@{event.x},{event.y}")
        except Exception:
            return
        if self.console_tabs.tabs()[idx] == str(self.plus_tab):
            return
        self._close_tab_by_index(idx)

    def _close_active_tab(self):
        try:
            sel = self.console_tabs.select()
            if sel and sel != str(self.plus_tab):
                idx = self.console_tabs.index(sel)
                self._close_tab_by_index(idx)
        except Exception:
            pass

    def _close_tab_by_index(self, idx):
        try:
            tab_id = self.console_tabs.tabs()[idx]
        except Exception:
            return
        for c in list(self.consoles):
            if str(c["frame"]) == tab_id:
                self._close_console_tab(c)
                break

    def _on_tab_change(self, event=None):
        # Keep "+" at the end
        try:
            idx_plus = self.console_tabs.index(self.plus_tab)
            last = len(self.console_tabs.tabs()) - 1
            if idx_plus != last:
                self.console_tabs.forget(self.plus_tab)
                self.console_tabs.add(self.plus_tab, text="+")
        except Exception:
            pass

    def _maybe_create_from_plus(self, event):
        try:
            idx = self.console_tabs.index("@%d,%d" % (event.x, event.y))
            if self.console_tabs.tabs()[idx] == str(self.plus_tab):
                self.new_console_tab()
        except Exception:
            pass

    # ---------- File Browser ----------
    def _build_file_browser(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Files").pack(side=tk.LEFT, padx=(2,6))
        ttk.Button(top, text="Root...", command=self._set_fb_root).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Refresh", command=self._fb_refresh).pack(side=tk.LEFT, padx=2)

        self.fb_search_var = tk.StringVar()
        self.fb_search_entry = ttk.Entry(top, textvariable=self.fb_search_var, width=15)
        self.fb_search_entry.pack(side=tk.RIGHT, padx=(6,2))
        self.fb_search_var.trace("w", self._fb_filter_tree)

        self.fb_tree = ttk.Treeview(parent, columns=("fullpath", "type"), displaycolumns=(), show="tree")
        self.fb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fb_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.fb_tree.yview)
        fb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.fb_tree.configure(yscrollcommand=fb_scroll.set)

        self.fb_tree.bind("<Button-1>", self._fb_left_click_menu)
        self.fb_tree.bind("<Double-1>", self._fb_open_item)
        self.fb_tree.bind("<<TreeviewOpen>>", self._fb_on_open)

        self.fb_menu = tk.Menu(self, tearoff=False)
        self.fb_menu.add_command(label="Open", command=lambda: self._fb_open_item(None))
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Duplicate", command=self._fb_duplicate)
        self.fb_menu.add_command(label="Rename...", command=self._fb_rename)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="New Folder", command=self._fb_new_folder)
        self.fb_menu.add_command(label="New File", command=self._fb_new_file)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Copy", command=lambda: self._fb_copy(mode="copy"))
        self.fb_menu.add_command(label="Move", command=lambda: self._fb_copy(mode="move"))
        self.fb_menu.add_command(label="Paste", command=self._fb_paste)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Delete", command=self._fb_delete)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Refresh", command=self._fb_refresh)

        self._init_fb_root()

    def _fb_filter_tree(self, *args):
        query = self.fb_search_var.get().lower()
        self._fb_rebuild_tree(filter_query=query)

    def _fb_rebuild_tree(self, filter_query=""):
        self.fb_tree.delete(*self.fb_tree.get_children())
        if not self.fb_root: return
        root_id = self.fb_tree.insert("", "end", text=str(self.fb_root.name), open=True, values=(str(self.fb_root), "dir"))
        self._fb_populate_dir(root_id, self.fb_root, filter_query)

    def _fb_populate_dir(self, parent_id, path: Path, filter_query=""):
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as e:
            messagebox.showerror("Read error", f"Cannot read {path}: {e}")
            return
        for e in entries:
            if filter_query and filter_query not in e.name.lower():
                continue
            node = self.fb_tree.insert(parent_id, "end", text=e.name, open=False,
                                       values=(str(e), "dir" if e.is_dir() else "file"))
            if e.is_dir():
                self.fb_tree.insert(node, "end", text=".", values=("", "dummy"))

    def _init_fb_root(self):
        if self.cfg.get("fb_root"):
            p = Path(self.cfg["fb_root"])
            if p.exists():
                self.fb_root = p
        if not self.fb_root:
            self.fb_root = self.project_root or Path.cwd()
        self._fb_rebuild_tree()

    def _set_fb_root(self):
        d = filedialog.askdirectory(title="Choose File Browser Root")
        if not d: return
        p = Path(d)
        if not p.exists():
            messagebox.showerror("Invalid folder", "That folder does not exist.")
            return
        self.fb_root = p
        self.cfg["fb_root"] = str(p)
        save_cfg(self.cfg)
        self._fb_rebuild_tree()

    def _fb_on_open(self, event):
        item = self.fb_tree.focus()
        if not item: return
        fullpath = Path(self.fb_tree.set(item, "fullpath"))
        if fullpath.is_dir():
            for child in self.fb_tree.get_children(item):
                self.fb_tree.delete(child)
            self._fb_populate_dir(item, fullpath, self.fb_search_var.get().lower())

    def _fb_left_click_menu(self, event):
        iid = self.fb_tree.identify_row(event.y)
        if iid:
            self.fb_tree.selection_set(iid)
            self.after(1, lambda: self.fb_menu.tk_popup(event.x_root, event.y_root))

    def _fb_selected_path(self) -> Path | None:
        sel = self.fb_tree.selection()
        if not sel: return None
        return Path(self.fb_tree.set(sel[0], "fullpath"))

    def _fb_selected_dir_for_new(self) -> Path | None:
        p = self._fb_selected_path()
        if not p: return None
        return p if p.is_dir() else p.parent

    def _fb_open_item(self, event=None):
        p = self._fb_selected_path()
        if not p or p.is_dir(): return
        if not self._maybe_save_changes(): return
        try:
            data = p.read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = p
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)

    def _fb_refresh(self):
        self._fb_rebuild_tree()

    def _fb_new_folder(self):
        base = self._fb_selected_dir_for_new()
        if not base:
            messagebox.showinfo("New Folder", "Select a folder first.")
            return
        name = simpledialog.askstring("New Folder", "Folder name:")
        if not name: return
        target = base / name
        try:
            target.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            messagebox.showerror("Exists", "Name already exists.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create folder:\n{e}")
            return
        self._fb_rebuild_tree()

    def _fb_new_file(self):
        base = self._fb_selected_dir_for_new()
        if not base:
            messagebox.showinfo("New File", "Select a folder first.")
            return
        name = simpledialog.askstring("New File", "File name (include extension):")
        if not name: return
        target = base / name
        if target.exists():
            messagebox.showerror("Exists", "Name already exists.")
            return
        try:
            target.write_text("", encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create file:\n{e}")
            return
        self._fb_rebuild_tree()

    def _fb_duplicate(self):
        p = self._fb_selected_path()
        if not p: return
        try:
            if p.is_dir():
                dst = self._unique_name(p.parent, p.name + "_copy")
                shutil.copytree(p, dst)
            else:
                stem = p.stem
                suffix = p.suffix
                dst = self._unique_name(p.parent, f"{stem}_copy{suffix}")
                shutil.copy2(p, dst)
        except Exception as e:
            messagebox.showerror("Duplicate failed", f"{e}")
            return
        self._fb_rebuild_tree()

    def _unique_name(self, parent: Path, name: str) -> Path:
        candidate = parent / name
        if not candidate.exists():
            return candidate
        stem, suffix = os.path.splitext(name)
        n = 2
        while True:
            candidate = parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def _fb_rename(self):
        p = self._fb_selected_path()
        if not p: return
        new_name = simpledialog.askstring("Rename", "New name:", initialvalue=p.name)
        if not new_name or new_name == p.name: return
        new_path = p.parent / new_name
        if new_path.exists():
            messagebox.showerror("Exists", "Target name already exists.")
            return
        try:
            p.rename(new_path)
        except Exception as e:
            messagebox.showerror("Rename failed", f"{e}")
            return
        self._fb_rebuild_tree()
        if self.filename and self.filename == p:
            self.filename = new_path
            self._update_title_status()

    def _fb_copy(self, mode="copy"):
        paths = []
        for iid in self.fb_tree.selection():
            fp = self.fb_tree.set(iid, "fullpath")
            if fp: paths.append(Path(fp))
        if not paths:
            p = self._fb_selected_path()
            if p: paths = [p]
        if not paths: return
        self.fs_clipboard = {"mode": mode, "paths": [str(x) for x in paths]}
        self.status.config(text=f"[{mode.UPPER()} set] {len(paths)} item(s)")

    def _fb_paste(self):
        if not self.fs_clipboard["paths"]:
            self.status.config(text="[Paste] Clipboard empty")
            return
        dest_dir = self._fb_selected_dir_for_new()
        if not dest_dir:
            messagebox.showinfo("Paste", "Select a destination folder first.")
            return
        mode = self.fs_clipboard["mode"]
        errors = []
        for spath in self.fs_clipboard["paths"]:
            sp = Path(spath)
            try:
                if mode == "copy":
                    if sp.is_dir():
                        dst = self._unique_name(dest_dir, sp.name)
                        shutil.copytree(sp, dst)
                    else:
                        dst = self._unique_name(dest_dir, sp.name)
                        shutil.copy2(sp, dst)
                elif mode == "move":
                    dst = dest_dir / sp.name
                    if dst.exists():
                        dst = self._unique_name(dest_dir, sp.name)
                    shutil.move(str(sp), str(dst))
            except Exception as e:
                errors.append(f"{sp.name}: {e}")
        self._fb_rebuild_tree()
        if errors:
            messagebox.showerror("Paste finished with errors", "\n".join(errors))
        else:
            self.status.config(text=f"[{mode.upper()}] Done")

    def _fb_delete(self):
        p = self._fb_selected_path()
        if not p: return
        if not messagebox.askyesno("Delete", f"Delete '{p.name}'?\nThis cannot be undone."):
            return
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            messagebox.showerror("Delete failed", f"{e}")
            return
        self._fb_rebuild_tree()

    # ---------- Pins ----------
    def _pins(self):
        return self.cfg.get("pins") or [None]*5

    def _save_pins(self, pins):
        self.cfg["pins"] = pins
        save_cfg(self.cfg)
        self._update_pin_button_labels()

    def _pin_button_click(self, idx: int, event):
        if event.state & 0x0001:  # Shift
            self._pin_set(idx)
        elif event.state & 0x0004:  # Control
            self._pin_clear(idx)
        else:
            self._pin_open(idx)

    def _pin_open(self, idx: int):
        pins = self._pins()
        p = pins[idx]
        if not p:
            self.status.config(text=f"[Pin {idx+1}] Empty")
            return
        path = Path(p)
        if not path.exists():
            messagebox.showwarning("Missing", f"Pinned file not found:\n{p}\n(Clearing this pin.)")
            pins[idx] = None
            self._save_pins(pins)
            return
        if not self._maybe_save_changes(): return
        try:
            data = path.read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = path
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)

    def _pin_set(self, idx: int):
        if not self.filename:
            messagebox.showinfo("Pin", "Save/open a file first.")
            return
        pins = self._pins()
        pins[idx] = str(self.filename)
        self._save_pins(pins)
        self.status.config(text=f"[Pin {idx+1}] Set to {self.filename.name}")

    def _pin_clear(self, idx: int):
        pins = self._pins()
        pins[idx] = None
        self._save_pins(pins)
        self.status.config(text=f"[Pin {idx+1}] Cleared")

    def _update_pin_button_labels(self):
        pins = self._pins()
        for i, btn in enumerate(self.pin_buttons):
            btn.configure(text=f"Pin{i+1}")

    # ---------- Bind keys ----------
    def _bind_keys(self):
        self.bind_all("<Control-s>", lambda e: (self.save_file(), "break"))
        self.bind_all("<Control-o>", lambda e: (self.open_file(), "break"))
        self.bind_all("<Control-n>", lambda e: (self.new_file(), "break"))
        self.bind_all("<Control-a>", self._select_all)
        self.bind_all("<Control-z>", lambda e: (self.text.event_generate("<<Undo>>"), "break"))
        self.bind_all("<Control-y>", lambda e: (self.text.event_generate("<<Redo>>"), "break"))
        self.bind_all("<F5>", lambda e: (self.run_current(), "break"))
        self.bind_all("<Control-Shift-D>", lambda e: (self.toggle_dark_mode(), "break"))
        self.bind_all("<Control-w>", lambda e: (self._close_active_tab(), "break"))

    # ---------- Theme ----------
    def toggle_dark_mode(self):
        self.cfg["dark_mode"] = not self.cfg.get("dark_mode", True)
        save_cfg(self.cfg)
        self.theme = DARK if self.cfg["dark_mode"] else LIGHT
        self._apply_theme()
        self._schedule_line_numbers()

    def _apply_theme(self):
        t = self.theme
        # Root areas
        self.configure(bg=t["bg"])
        self.toolbar_outer.configure(bg=t["toolbar_bg"])
        self.status_outer.configure(bg=t["status_bg"])
        self.status.configure(background=t["status_bg"], foreground=t["status_fg"])

        # Text + gutter + search highlight
        self.text.configure(bg=t["bg"], fg=t["fg"], insertbackground=t["cursor"],
                            selectbackground=t["sel_bg"], selectforeground=t["sel_fg"])
        self.text.tag_configure("search_hit", background=t["search_bg"], foreground=t["search_fg"])
        self.gutter_canvas.configure(bg=t["gutter_bg"])

        # Menus
        self.option_add("*Menu.background", t["menu_bg"])
        self.option_add("*Menu.foreground", t["menu_fg"])
        self.option_add("*Menu.activeBackground", t["menu_active"])
        self.option_add("*Menu.activeForeground", t["menu_fg"])

        # ttk styling
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background=t["bg"])
        style.configure("Tool.TFrame", background=t["toolbar_bg"])
        self.toolbar.configure(style="Tool.TFrame")

        style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        style.configure("TButton", background=t["toolbar_bg"], foreground=t["fg"], padding=6)
        style.map("TButton",
                  background=[("active", t["tab_sel_bg"])],
                  foreground=[("active", t["tab_sel_fg"])])

        style.configure("TEntry", fieldbackground=t["bg"], foreground=t["fg"])
        style.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        style.configure("Vertical.TScrollbar", background=t["bg"])
        style.configure("Horizontal.TScrollbar", background=t["bg"])

        style.configure("TNotebook", background=t["bg"])
        style.configure("TNotebook.Tab", background=t["tab_bg"], foreground=t["tab_fg"], padding=(10, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", t["tab_sel_bg"])],
                  foreground=[("selected", t["tab_sel_fg"])])

        style.configure('Treeview',
                        background=t['bg'], foreground=t['fg'],
                        fieldbackground=t['bg'], borderwidth=0)
        style.map('Treeview',
                  background=[('selected', t['tree_sel_bg'])],
                  foreground=[('selected', t['tree_sel_fg'])])
        style.configure('Treeview.Heading',
                        background=t['toolbar_bg'], foreground=t['fg'])

    # ---------- File ops ----------
    def new_file(self):
        if not self._maybe_save_changes(): return
        self.filename = None
        self.text.delete("1.0", tk.END)
        self._modified = False
        self._update_title_status()
        self._schedule_line_numbers()

    def open_file(self):
        if not self._maybe_save_changes(): return
        path = filedialog.askopenfilename(title="Open file")
        if not path: return
        try:
            data = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = Path(path)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)
        if not self.cfg.get("fb_root"):
            self.fb_root = self.project_root or self.fb_root
            self._fb_rebuild_tree()
        self._schedule_line_numbers()

    def save_file(self):
        if self.filename is None:
            return self.save_file_as()
        try:
            data = self.text.get("1.0", tk.END)
            self.filename.write_text(data.rstrip("\n") + "\n", encoding="utf-8")
            self._modified = False
            self._update_title_status()
            self.status.config(text=f"Saved to {self.filename}")
            self.project_root = self._find_pkg_root(self.filename.parent)
            return True
        except Exception as e:
            messagebox.showerror("Save failed", f"{e}")
            return False

    def save_file_as(self):
        path = filedialog.asksaveasfilename(title="Save As", defaultextension="",
                                            initialfile=(self.filename.name if self.filename else "untitled.txt"))
        if not path: return False
        self.filename = Path(path)
        self.project_root = self._find_pkg_root(self.filename.parent)
        return self.save_file()

    def restart_app(self):
        if not self._maybe_save_changes():
            return
        try:
            for c in self.consoles:
                try:
                    if c["proc"]:
                        c["proc"].terminate()
                except Exception:
                    pass
        except Exception:
            pass
        python = sys.executable
        args = [python] + sys.argv
        self.destroy()
        os.execv(python, args)

    # ---------- Edit ops ----------
    def _select_all(self, event=None):
        self.text.tag_add(tk.SEL, "1.0", tk.END)
        self.text.mark_set(tk.INSERT, "1.0")
        self.text.see(tk.INSERT)
        return "break"

    def copy_all(self):
        data = self.text.get("1.0", tk.END).rstrip()
        self.clipboard_clear()
        self.clipboard_append(data)
        self.status.config(text="[All code copied]")
        self.after(1500, self._update_title_status)

    # ---------- Run integration ----------
    def run_current(self):
        if self.filename is None:
            choice = messagebox.askyesnocancel(
                "Save file?",
                "This buffer has never been saved.\n"
                "Yes = Save As and run\n"
                "No = Run a temp copy\n"
                "Cancel = Abort"
            )
            if choice is None:
                return
            if choice:
                if not self.save_file_as(): return
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as tmpf:
                    tmpf.write(self.text.get("1.0", tk.END))
                    tmp_path = Path(tmpf.name)
                self._start_process_for_path(tmp_path, temp_mode=True)
                return
        else:
            if self._modified and not self.save_file():
                return
        self._start_process_for_path(self.filename, temp_mode=False)

    def _start_process_for_path(self, path: Path, temp_mode: bool):
        cmd = self._compute_run_cmd_for(path)
        con = self._current_console()
        if not con:
            self.new_console_tab()
            con = self._current_console()
        self._console_clear(con)
        self._console_write(con, f"$ {cmd}\n")
        try:
            con["proc"] = subprocess.Popen(
                cmd, shell=True, cwd=str(path.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True, bufsize=1,
                creationflags=CREATE_FLAGS
            )
            self.btn_run.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)
            threading.Thread(target=self._pump_output, args=(con,), daemon=True).start()
        except Exception as e:
            self._console_write(con, f"[error] {e}\n")
        if temp_mode:
            def cleanup():
                if con["proc"] is None:
                    try: path.unlink(missing_ok=True)
                    except Exception: pass
                else:
                    self.after(500, cleanup)
            self.after(500, cleanup)

    def stop_run(self):
        con = self._current_console()
        if not con: return
        p = con.get("proc")
        if not p: return
        try:
            p.terminate()
            self._console_write(con, "\n[process terminated]\n")
        except Exception as e:
            self._console_write(con, f"\n[stop error] {e}\n")
        finally:
            con["proc"] = None
            self.btn_run.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)

    def _pump_output(self, con):
        p = con["proc"]
        try:
            for line in p.stdout:
                self._console_write(con, line)
        except Exception as e:
            self._console_write(con, f"\n[stream error] {e}\n")
        finally:
            rc = p.wait()
            self._console_write(con, f"\n[process exited with code {rc}]\n")
            con["proc"] = None
            self.after(0, lambda: (self.btn_run.configure(state=tk.NORMAL), self.btn_stop.configure(state=tk.DISABLED)))

    def _compute_run_cmd_for(self, path: Path) -> str:
        # Always prefer PowerShell; no raw cmd.exe
        ext = path.suffix.lower()
        if ext == ".ps1": return f'powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "{path}"'
        if ext == ".py":  return f'powershell -NoLogo -NoProfile -Command python "{path}"'
        if ext == ".js":  return f'powershell -NoLogo -NoProfile -Command node "{path}"'
        if ext in (".bat", ".cmd"): return f'powershell -NoLogo -NoProfile -Command "& \'{path}\'"'
        if os.access(path, os.X_OK): return f'powershell -NoLogo -NoProfile -Command "& \'{path}\'"'
        return f'powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "{path}"'

    # ---------- Console tabs (PowerShell) ----------
    def new_console_tab(self):
        idx_plus = self.console_tabs.index(self.plus_tab)
        frame = ttk.Frame(self.console_tabs)

        # Top: output
        top = ttk.Frame(frame)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        text = tk.Text(top, height=12, wrap="word", state=tk.DISABLED, font=MONO_FONT)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(top, orient="vertical", command=text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll.set)

        # Bottom: dual input bars
        bottom = ttk.Frame(frame, padding=(0,6))
        bottom.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bottom, text="Command:").grid(row=0, column=0, padx=(0,6))
        cmd_entry = ttk.Entry(bottom)
        cmd_entry.grid(row=0, column=1, sticky="ew", padx=(0,6))
        cmd_btn = ttk.Button(bottom, text="Send", command=lambda: self._send_command(console))
        cmd_btn.grid(row=0, column=2)

        ttk.Label(bottom, text="Prompt:").grid(row=1, column=0, padx=(0,6), pady=(6,0))
        prompt_entry = ttk.Entry(bottom)
        prompt_entry.grid(row=1, column=1, sticky="ew", padx=(0,6), pady=(6,0))
        prompt_btn = ttk.Button(bottom, text="Send", command=lambda: self._send_prompt(console))
        prompt_btn.grid(row=1, column=2, pady=(6,0))

        bottom.columnconfigure(1, weight=1)

        # Launch PowerShell (hidden window)
        try:
            proc = subprocess.Popen(
                "powershell -NoLogo -NoProfile",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=CREATE_FLAGS
            )
        except Exception as e:
            proc = None
            text.configure(state=tk.NORMAL)
            text.insert(tk.END, f"[Error launching PowerShell: {e}]\n")
            text.configure(state=tk.DISABLED)

        console = {"frame": frame, "text": text, "cmd_entry": cmd_entry, "prompt_entry": prompt_entry, "proc": proc}
        self.consoles.append(console)

        # Events
        cmd_entry.bind("<Return>", lambda e: (self._send_command(console), "break"))
        prompt_entry.bind("<Return>", lambda e: (self._send_prompt(console), "break"))

        # Theme on text area
        t = self.theme
        text.configure(bg=t["console_bg"], fg=t["console_fg"], insertbackground=t["cursor"])

        # Insert before "+" and re-add "+" to end
        title = "× PowerShell"
        self.console_tabs.insert(idx_plus, frame, text=title)
        self.console_tabs.forget(self.plus_tab)
        self.console_tabs.add(self.plus_tab, text="+")
        self.console_tabs.select(frame)

        if proc:
            threading.Thread(target=self._pump_console_output, args=(console,), daemon=True).start()

    def _current_console(self):
        try:
            sel = self.console_tabs.select()
            for c in self.consoles:
                if str(c["frame"]) == sel:
                    return c
        except Exception:
            return None
        return None

    def _pump_console_output(self, console):
        p = console["proc"]
        try:
            for line in p.stdout:
                self._console_write(console, line)
        except Exception as e:
            self._console_write(console, f"\n[stream error] {e}\n")
        finally:
            try:
                rc = p.wait()
            except Exception:
                rc = "?"
            self._console_write(console, f"\n[console exited with code {rc}]\n")

    def _send_command(self, console):
        data = console["cmd_entry"].get().strip()
        console["cmd_entry"].delete(0, tk.END)
        if not data: return
        self._console_write(console, f"> {data}\n")
        p = console.get("proc")
        if not p or not p.stdin:
            self._console_write(console, "[no console process]\n")
            return
        try:
            p.stdin.write(data + "\n")
            p.stdin.flush()
        except Exception as ex:
            self._console_write(console, f"[input error] {ex}\n")

    def _send_prompt(self, console):
        data = console["prompt_entry"].get()
        console["prompt_entry"].delete(0, tk.END)
        if data == "": return
        self._console_write(console, data + "\n")

    def _close_console_tab(self, console):
        try:
            if console.get("proc"):
                try:
                    console["proc"].terminate()
                except Exception:
                    pass
        finally:
            try:
                self.console_tabs.forget(console["frame"])
            except Exception:
                pass
            self.consoles = [c for c in self.consoles if c is not console]

    def _console_clear(self, console):
        t = console["text"]
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        t.configure(state=tk.DISABLED)

    def _console_write(self, console, text: str):
        t = console["text"]
        t.configure(state=tk.NORMAL)
        t.insert(tk.END, text)
        t.see(tk.END)
        t.configure(state=tk.DISABLED)

    # ---------- Scrolling / line numbers ----------
    def _yview_text(self, *args):
        self.text.yview(*args)
        self._schedule_line_numbers()

    def _yscrollcommand(self, sb, first, last):
        sb.set(first, last)
        self._schedule_line_numbers()

    def _schedule_line_numbers(self):
        # Throttle: draw once per idle cycle
        if getattr(self, "_ln_sched", False): return
        self._ln_sched = True
        self.after_idle(self._update_line_numbers)

    def _update_line_numbers(self):
        # Guard against early callbacks during teardown/early init
        if not self.winfo_exists() or not self.text.winfo_exists() or not self.gutter_canvas.winfo_exists():
            return
        self._ln_sched = False
        t = self.theme
        canvas = self.gutter_canvas
        canvas.delete("all")

        total_lines = int(self.text.index("end-1c").split(".")[0])
        digits = max(3, len(str(total_lines)))
        char_w = self._ln_font.measure("9")
        pad = 10
        width_px = digits * char_w + pad
        canvas.configure(width=width_px, bg=t["gutter_bg"])

        index = self.text.index("@0,0")
        height = self.text.winfo_height()
        fg = t["gutter_fg"]

        while True:
            dli = self.text.dlineinfo(index)
            if dli is None:
                break
            y = dli[1]
            if y > height:
                break
            line_no = int(index.split(".")[0])
            canvas.create_text(width_px - 6, y, anchor="ne", text=str(line_no), fill=fg, font=self._ln_font)
            index = self.text.index(f"{line_no + 1}.0")

    # ---------- Helpers ----------
    def _update_title_status(self):
        name = (self.filename.name if self.filename else "untitled")
        mod = " *" if self._modified else ""
        root_hint = f"  {self.project_root}" if self.project_root else ""
        self.title(f"{APP_TITLE} - {name}{mod}")
        idx = self.text.index(tk.INSERT)
        line, col = idx.split(".")
        self.status.config(text=f"{name}{mod}{root_hint}   |   Line {line}, Col {int(col)+1}")

    def _on_modified(self, event=None):
        if self.text.edit_modified():
            self._modified = True
            self.text.edit_modified(False)
        self._update_title_status()
        self._schedule_line_numbers()

    def _on_close(self):
        if not self._maybe_save_changes(): return
        try:
            for c in self.consoles:
                try:
                    if c["proc"]:
                        c["proc"].terminate()
                except Exception:
                    pass
        finally:
            self.destroy()

    def _maybe_save_changes(self) -> bool:
        if not self._modified:
            return True
        resp = messagebox.askyesnocancel("Unsaved changes", "Save changes before closing?")
        if resp is None: return False
        if resp: return bool(self.save_file())
        return True

    # ---------- Project helpers ----------
    def _find_pkg_root(self, start_dir: Path | None) -> Path | None:
        if not start_dir:
            return None
        d = start_dir.resolve()
        for _ in range(30):
            if (d / "package.json").exists():
                return d
            if d.parent == d:
                break
            d = d.parent
        return None

    def _set_project_root(self):
        d = filedialog.askdirectory(title="Choose Project Root")
        if not d:
            return
        self.project_root = Path(d)
        self.status.config(text=f"Project root set: {self.project_root}")

if __name__ == "__main__":
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = MiniNano()
    app.mainloop()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniNano (Windows-only) — unified console, robust line numbers, full dark mode, inline Ctrl+F

- Windows only (all *nix code removed)
- One PowerShell console per tab (no separate GPT tab)
- Two input bars per console tab:
    • Command -> executes in PowerShell
    • Prompt  -> prints text to console (no exec)
- Browser-style "+" tab for new consoles; close tab by clicking the left ~16px,
  middle-click, or Ctrl+W
- Canvas-based line-number gutter (no flicker or missing lines)
- Dark theme applied to editor, tree, tabs, buttons, entries, scrollbars, menus, status bar
- Inline Find bar (Ctrl+F): next/prev, count; Enter=next, Shift+Enter=prev, F3/Shift+F3 too
- All PowerShell/background processes run hidden (no stray cmd windows)
"""
import os, sys, threading, subprocess, json, shutil, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import tkinter.font as tkfont

if os.name != "nt":
    raise SystemExit("This build of MiniNano is Windows-only.")

APP_TITLE = "MiniNano"
MONO_FONT = ("Consolas", 11)

# ---------- Config ----------
CFG_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "mininano"
CFG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULTS = {
    "dark_mode": True,
    "pins": [None, None, None, None, None],
    "fb_root": None
}
CFG_PATH = CFG_DIR / "config.json"

def load_cfg():
    if CFG_PATH.exists():
        try:
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return DEFAULTS.copy()

def save_cfg(cfg: dict):
    try:
        CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------- Themes ----------
LIGHT = {
    "bg": "#ffffff", "fg": "#1e1e1e", "sel_bg": "#cde8ff", "sel_fg": "#000000",
    "cursor": "#222222", "console_bg": "#fafafa", "console_fg": "#111111",
    "toolbar_bg": "#e9e9e9", "status_bg": "#efefef", "status_fg": "#333333",
    "search_bg": "#fff7a8", "search_fg": "#000000",
    "gutter_bg": "#f3f3f3", "gutter_fg": "#666666",
    "menu_bg": "#ffffff", "menu_fg": "#1e1e1e", "menu_active": "#e9e9e9",
    "tab_bg": "#e3e3e3", "tab_fg": "#1e1e1e", "tab_sel_bg": "#d7d7d7", "tab_sel_fg": "#000000",
    "tree_sel_bg": "#cde8ff", "tree_sel_fg": "#000000",
}
DARK = {
    "bg": "#1e1e1e", "fg": "#d4d4d4", "sel_bg": "#264f78", "sel_fg": "#ffffff",
    "cursor": "#cccccc", "console_bg": "#111317", "console_fg": "#d6dde6",
    "toolbar_bg": "#2b2d31", "status_bg": "#2b2d31", "status_fg": "#c7c7c7",
    "search_bg": "#3a3d41", "search_fg": "#ffd866",
    "gutter_bg": "#252526", "gutter_fg": "#8e8e8e",
    "menu_bg": "#1e1e1e", "menu_fg": "#d4d4d4", "menu_active": "#2b2d31",
    "tab_bg": "#2b2d31", "tab_fg": "#cfd3da", "tab_sel_bg": "#3a3d41", "tab_sel_fg": "#ffffff",
    "tree_sel_bg": "#264f78", "tree_sel_fg": "#ffffff",
}

CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

class MiniNano(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x820")
        self.minsize(900, 560)

        # State
        self.cfg = load_cfg()
        self.theme = DARK if self.cfg.get("dark_mode", True) else LIGHT
        self.filename: Path | None = None
        self._modified = False
        self.project_root: Path | None = None
        self.fb_root: Path | None = None
        self.fs_clipboard = {"mode": None, "paths": []}
        self.consoles = []  # [{frame, text, cmd_entry, prompt_entry, proc}]
        self._ln_font = tkfont.Font(master=self, font=MONO_FONT)  # gutter font (explicit master)
        # inline find state
        self.find_term = ""
        self.find_matches = []
        self.find_index = -1

        # UI
        self._build_ui()
        self._apply_theme()
        self._bind_keys()
        self.new_file()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        # Menu
        menubar = tk.Menu(self)
        file_m = tk.Menu(menubar, tearoff=False)
        file_m.add_command(label="New", accelerator="Ctrl+N", command=self.new_file)
        file_m.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_file)
        file_m.add_separator()
        file_m.add_command(label="Save", accelerator="Ctrl+S", command=self.save_file)
        file_m.add_command(label="Save As...", command=self.save_file_as)
        file_m.add_separator()
        file_m.add_command(label="Run (F5)", accelerator="F5", command=self.run_current)
        file_m.add_command(label="Stop", accelerator="Esc", command=self.stop_run)
        file_m.add_separator()
        file_m.add_command(label="Restart", command=self.restart_app)
        file_m.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_m)

        project_m = tk.Menu(menubar, tearoff=False)
        project_m.add_command(label="Set Project Root...", command=self._set_project_root)
        project_m.add_command(label="Set File Browser Root...", command=self._set_fb_root)
        project_m.add_command(label="Refresh File Browser", command=self._fb_refresh)
        menubar.add_cascade(label="Project", menu=project_m)

        view_m = tk.Menu(menubar, tearoff=False)
        self.dark_mode_var = tk.BooleanVar(value=self.cfg.get("dark_mode", True))
        view_m.add_checkbutton(label="Dark Mode", onvalue=True, offvalue=False,
                               variable=self.dark_mode_var, command=self.toggle_dark_mode,
                               accelerator="Ctrl+Shift+D")
        menubar.add_cascade(label="View", menu=view_m)

        help_m = tk.Menu(menubar, tearoff=False)
        help_m.add_command(label="About", command=lambda: messagebox.showinfo(
            "About MiniNano",
            "MiniNano — Windows PowerShell console with browser-style tabs, two input bars (Command & Prompt)."
        ))
        menubar.add_cascade(label="Help", menu=help_m)
        self.config(menu=menubar)

        # Toolbar
        self.toolbar_outer = tk.Frame(self, bd=0, highlightthickness=0)
        self.toolbar_outer.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = ttk.Frame(self.toolbar_outer, padding=(6, 6))
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_open = ttk.Button(self.toolbar, text="Open", command=self.open_file)
        self.btn_save = ttk.Button(self.toolbar, text="Save (Ctrl+S)", command=self.save_file)
        self.btn_save_as = ttk.Button(self.toolbar, text="Save As...", command=self.save_file_as)
        self.btn_undo = ttk.Button(self.toolbar, text="Undo (Ctrl+Z)", command=lambda: self.text.event_generate("<<Undo>>"))
        self.btn_redo = ttk.Button(self.toolbar, text="Redo (Ctrl+Y)", command=lambda: self.text.event_generate("<<Redo>>"))
        self.btn_run = ttk.Button(self.toolbar, text="Run (F5)", command=self.run_current)
        self.btn_stop = ttk.Button(self.toolbar, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.btn_copyall = ttk.Button(self.toolbar, text="Copy All", command=self.copy_all)
        self.btn_restart = ttk.Button(self.toolbar, text="Restart", command=self.restart_app)
        for w in (self.btn_open, self.btn_save, self.btn_save_as, self.btn_undo, self.btn_redo,
                  self.btn_run, self.btn_stop, self.btn_copyall, self.btn_restart):
            w.pack(side=tk.LEFT, padx=(0, 6))

        # Pin buttons
        self.pin_buttons_frame = ttk.Frame(self.toolbar)
        self.pin_buttons_frame.pack(side=tk.RIGHT)
        self.pin_buttons = []
        for i in range(5):
            btn = ttk.Button(self.pin_buttons_frame, text=f"Pin{i+1}", width=6)
            btn.bind("<Button-1>", lambda e, i=i: self._pin_button_click(i, e))
            btn.pack(side=tk.LEFT, padx=2)
            self.pin_buttons.append(btn)
        self._update_pin_button_labels()

        # Main layout
        self.root_paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.root_paned.pack(fill=tk.BOTH, expand=True)

        # Top: File Browser | Editor
        self.top_paned = ttk.PanedWindow(self.root_paned, orient=tk.HORIZONTAL)
        self.root_paned.add(self.top_paned, weight=5)

        # --- File Browser
        self.fb_panel = ttk.Frame(self.top_paned, width=260)
        self._build_file_browser(self.fb_panel)
        self.top_paned.add(self.fb_panel, weight=1)

        # --- Editor + gutter (canvas-based line numbers)
        editor_outer = ttk.Frame(self.top_paned)
        self.top_paned.add(editor_outer, weight=5)

        self.gutter_canvas = tk.Canvas(editor_outer, width=40, highlightthickness=0, bd=0)
        self.gutter_canvas.pack(side=tk.LEFT, fill=tk.Y)

        editor_frame = ttk.Frame(editor_outer)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.text = tk.Text(editor_frame, wrap="none", undo=True, font=MONO_FONT, tabs=("1c",), insertwidth=2)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(editor_frame, orient="vertical", command=self._yview_text)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=lambda *a: self._yscrollcommand(yscroll, *a))

        # Search highlight tag
        self.text.tag_configure("search_hit")

        # --- Inline Find Bar (hidden by default; inline, not a popup)
        self.find_bar = ttk.Frame(editor_outer, padding=(6, 4))
        self.find_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.find_bar.pack_forget()

        ttk.Label(self.find_bar, text="Find:").pack(side=tk.LEFT, padx=(0, 6))
        self.find_entry = ttk.Entry(self.find_bar, width=32)
        self.find_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.find_prev_btn = ttk.Button(self.find_bar, text="◄ Prev", command=self._find_prev)
        self.find_prev_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.find_next_btn = ttk.Button(self.find_bar, text="Next ►", command=self._find_next)
        self.find_next_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.find_count = ttk.Label(self.find_bar, text="0/0")
        self.find_count.pack(side=tk.LEFT)

        self.find_entry.bind("<Return>", lambda e: (self._find_next(), "break"))
        self.find_entry.bind("<Shift-Return>", lambda e: (self._find_prev(), "break"))

        # Bottom: Console Notebook with "+" tab
        self.console_tabs = ttk.Notebook(self.root_paned)
        self.root_paned.add(self.console_tabs, weight=2)
        self._add_plus_tab()
        self._install_tab_close_behavior()
        self.new_console_tab()  # Start with one console

        # Status bar
        self.status_outer = tk.Frame(self, bd=0, highlightthickness=0)
        self.status_outer.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(self.status_outer, anchor="w", padding=(6, 3))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # Bindings
        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<KeyRelease>", lambda e: (self._update_title_status(), self._schedule_line_numbers()))
        self.text.bind("<ButtonRelease-1>", lambda e: (self._update_title_status(), self._schedule_line_numbers()))
        self.text.bind("<Configure>", lambda e: self._schedule_line_numbers())
        self.text.bind("<MouseWheel>", lambda e: self._schedule_line_numbers())

        self.console_tabs.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self.console_tabs.bind("<ButtonRelease-1>", self._maybe_create_from_plus)

    # ----- Inline Find (Ctrl+F) -----
    def _toggle_find(self):
        if self.find_bar.winfo_ismapped():
            self._close_find()
        else:
            self.find_bar.pack(side=tk.BOTTOM, fill=tk.X)
            # seed with current selection if any
            try:
                sel = self.text.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                sel = ""
            self.find_entry.delete(0, tk.END)
            if sel:
                self.find_entry.insert(0, sel)
            self.find_entry.focus_set()
            self._apply_theme()  # ensure colors match

    def _close_find(self):
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.find_bar.pack_forget()
        self.find_term = ""
        self.find_matches = []
        self.find_index = -1
        self.find_count.config(text="0/0")
        self.text.focus_set()

    def _find_all(self, term: str):
        self.find_matches = []
        if not term:
            return
        start = "1.0"
        while True:
            pos = self.text.search(term, start, stopindex=tk.END, nocase=1)
            if not pos:
                break
            end_pos = f"{pos}+{len(term)}c"
            self.find_matches.append((pos, end_pos))
            start = end_pos

    def _find_next(self):
        term = self.find_entry.get().strip()
        if not term:
            return
        if term != self.find_term:
            self.find_term = term
            self._find_all(term)
            self.find_index = -1
        if not self.find_matches:
            self.find_count.config(text="0/0"); return
        self.find_index = (self.find_index + 1) % len(self.find_matches)
        pos, endp = self.find_matches[self.find_index]
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.text.tag_add("search_hit", pos, endp)
        self.text.see(pos)
        self.find_count.config(text=f"{self.find_index + 1}/{len(self.find_matches)}")

    def _find_prev(self):
        term = self.find_entry.get().strip()
        if not term:
            return
        if term != self.find_term:
            self.find_term = term
            self._find_all(term)
            self.find_index = -1
        if not self.find_matches:
            self.find_count.config(text="0/0"); return
        if self.find_index == -1:
            self.find_index = len(self.find_matches) - 1
        else:
            self.find_index = (self.find_index - 1) % len(self.find_matches)
        pos, endp = self.find_matches[self.find_index]
        self.text.tag_remove("search_hit", "1.0", tk.END)
        self.text.tag_add("search_hit", pos, endp)
        self.text.see(pos)
        self.find_count.config(text=f"{self.find_index + 1}/{len(self.find_matches)}")

    # ---------- Console tabs "+" and close behavior ----------
    def _add_plus_tab(self):
        plus = ttk.Frame(self.console_tabs)
        self.console_tabs.add(plus, text="+")
        self.plus_tab = plus

    def _install_tab_close_behavior(self):
        # Click in left ~16px of a tab to close; middle-click closes; Ctrl+W closes active
        self.console_tabs.bind("<Button-1>", self._on_notebook_click, add="+")
        self.console_tabs.bind("<Button-2>", self._on_notebook_middle_click, add="+")
        self.console_tabs.bind("<Button-3>", self._on_notebook_middle_click, add="+")
        self.bind_all("<Control-w>", lambda e: (self._close_active_tab(), "break"))

        # Find shortcuts (inline)
        self.bind_all("<Control-f>", lambda e: (self._toggle_find(), "break"))
        self.bind_all("<F3>", lambda e: (self._find_next(), "break"))
        self.bind_all("<Shift-F3>", lambda e: (self._find_prev(), "break"))
        self.bind_all("<Escape>", self._handle_escape, add="+")

    def _handle_escape(self, event):
        if self.find_bar.winfo_ismapped():
            self._close_find()
            return "break"
        self.stop_run()
        return "break"

    def _on_notebook_click(self, event):
        try:
            idx = self.console_tabs.index(f"@{event.x},{event.y}")
        except Exception:
            return
        tab_id = self.console_tabs.tabs()[idx]
        if tab_id == str(self.plus_tab):
            return
        x, y, w, h = self.console_tabs.bbox(idx)
        if event.x <= x + 16:
            self._close_tab_by_index(idx)

    def _on_notebook_middle_click(self, event):
        try:
            idx = self.console_tabs.index(f"@{event.x},{event.y}")
        except Exception:
            return
        if self.console_tabs.tabs()[idx] == str(self.plus_tab):
            return
        self._close_tab_by_index(idx)

    def _close_active_tab(self):
        try:
            sel = self.console_tabs.select()
            if sel and sel != str(self.plus_tab):
                idx = self.console_tabs.index(sel)
                self._close_tab_by_index(idx)
        except Exception:
            pass

    def _close_tab_by_index(self, idx):
        try:
            tab_id = self.console_tabs.tabs()[idx]
        except Exception:
            return
        for c in list(self.consoles):
            if str(c["frame"]) == tab_id:
                self._close_console_tab(c)
                break

    def _on_tab_change(self, event=None):
        # Keep "+" at the end
        try:
            idx_plus = self.console_tabs.index(self.plus_tab)
            last = len(self.console_tabs.tabs()) - 1
            if idx_plus != last:
                self.console_tabs.forget(self.plus_tab)
                self.console_tabs.add(self.plus_tab, text="+")
        except Exception:
            pass

    def _maybe_create_from_plus(self, event):
        try:
            idx = self.console_tabs.index("@%d,%d" % (event.x, event.y))
            if self.console_tabs.tabs()[idx] == str(self.plus_tab):
                self.new_console_tab()
        except Exception:
            pass

    # ---------- File Browser ----------
    def _build_file_browser(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Files").pack(side=tk.LEFT, padx=(2,6))
        ttk.Button(top, text="Root...", command=self._set_fb_root).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Refresh", command=self._fb_refresh).pack(side=tk.LEFT, padx=2)

        self.fb_search_var = tk.StringVar()
        self.fb_search_entry = ttk.Entry(top, textvariable=self.fb_search_var, width=15)
        self.fb_search_entry.pack(side=tk.RIGHT, padx=(6,2))
        self.fb_search_var.trace("w", self._fb_filter_tree)

        self.fb_tree = ttk.Treeview(parent, columns=("fullpath", "type"), displaycolumns=(), show="tree")
        self.fb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fb_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.fb_tree.yview)
        fb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.fb_tree.configure(yscrollcommand=fb_scroll.set)

        self.fb_tree.bind("<Button-1>", self._fb_left_click_menu)
        self.fb_tree.bind("<Double-1>", self._fb_open_item)
        self.fb_tree.bind("<<TreeviewOpen>>", self._fb_on_open)

        self.fb_menu = tk.Menu(self, tearoff=False)
        self.fb_menu.add_command(label="Open", command=lambda: self._fb_open_item(None))
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Duplicate", command=self._fb_duplicate)
        self.fb_menu.add_command(label="Rename...", command=self._fb_rename)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="New Folder", command=self._fb_new_folder)
        self.fb_menu.add_command(label="New File", command=self._fb_new_file)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Copy", command=lambda: self._fb_copy(mode="copy"))
        self.fb_menu.add_command(label="Move", command=lambda: self._fb_copy(mode="move"))
        self.fb_menu.add_command(label="Paste", command=self._fb_paste)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Delete", command=self._fb_delete)
        self.fb_menu.add_separator()
        self.fb_menu.add_command(label="Refresh", command=self._fb_refresh)

        self._init_fb_root()

    def _fb_filter_tree(self, *args):
        query = self.fb_search_var.get().lower()
        self._fb_rebuild_tree(filter_query=query)

    def _fb_rebuild_tree(self, filter_query=""):
        self.fb_tree.delete(*self.fb_tree.get_children())
        if not self.fb_root: return
        root_id = self.fb_tree.insert("", "end", text=str(self.fb_root.name), open=True, values=(str(self.fb_root), "dir"))
        self._fb_populate_dir(root_id, self.fb_root, filter_query)

    def _fb_populate_dir(self, parent_id, path: Path, filter_query=""):
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as e:
            messagebox.showerror("Read error", f"Cannot read {path}: {e}")
            return
        for e in entries:
            if filter_query and filter_query not in e.name.lower():
                continue
            node = self.fb_tree.insert(parent_id, "end", text=e.name, open=False,
                                       values=(str(e), "dir" if e.is_dir() else "file"))
            if e.is_dir():
                self.fb_tree.insert(node, "end", text=".", values=("", "dummy"))

    def _init_fb_root(self):
        if self.cfg.get("fb_root"):
            p = Path(self.cfg["fb_root"])
            if p.exists():
                self.fb_root = p
        if not self.fb_root:
            self.fb_root = self.project_root or Path.cwd()
        self._fb_rebuild_tree()

    def _set_fb_root(self):
        d = filedialog.askdirectory(title="Choose File Browser Root")
        if not d: return
        p = Path(d)
        if not p.exists():
            messagebox.showerror("Invalid folder", "That folder does not exist.")
            return
        self.fb_root = p
        self.cfg["fb_root"] = str(p)
        save_cfg(self.cfg)
        self._fb_rebuild_tree()

    def _fb_on_open(self, event):
        item = self.fb_tree.focus()
        if not item: return
        fullpath = Path(self.fb_tree.set(item, "fullpath"))
        if fullpath.is_dir():
            for child in self.fb_tree.get_children(item):
                self.fb_tree.delete(child)
            self._fb_populate_dir(item, fullpath, self.fb_search_var.get().lower())

    def _fb_left_click_menu(self, event):
        iid = self.fb_tree.identify_row(event.y)
        if iid:
            self.fb_tree.selection_set(iid)
            self.after(1, lambda: self.fb_menu.tk_popup(event.x_root, event.y_root))

    def _fb_selected_path(self) -> Path | None:
        sel = self.fb_tree.selection()
        if not sel: return None
        return Path(self.fb_tree.set(sel[0], "fullpath"))

    def _fb_selected_dir_for_new(self) -> Path | None:
        p = self._fb_selected_path()
        if not p: return None
        return p if p.is_dir() else p.parent

    def _fb_open_item(self, event=None):
        p = self._fb_selected_path()
        if not p or p.is_dir(): return
        if not self._maybe_save_changes(): return
        try:
            data = p.read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = p
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)

    def _fb_refresh(self):
        self._fb_rebuild_tree()

    def _fb_new_folder(self):
        base = self._fb_selected_dir_for_new()
        if not base:
            messagebox.showinfo("New Folder", "Select a folder first.")
            return
        name = simpledialog.askstring("New Folder", "Folder name:")
        if not name: return
        target = base / name
        try:
            target.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            messagebox.showerror("Exists", "Name already exists.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create folder:\n{e}")
            return
        self._fb_rebuild_tree()

    def _fb_new_file(self):
        base = self._fb_selected_dir_for_new()
        if not base:
            messagebox.showinfo("New File", "Select a folder first.")
            return
        name = simpledialog.askstring("New File", "File name (include extension):")
        if not name: return
        target = base / name
        if target.exists():
            messagebox.showerror("Exists", "Name already exists.")
            return
        try:
            target.write_text("", encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create file:\n{e}")
            return
        self._fb_rebuild_tree()

    def _fb_duplicate(self):
        p = self._fb_selected_path()
        if not p: return
        try:
            if p.is_dir():
                dst = self._unique_name(p.parent, p.name + "_copy")
                shutil.copytree(p, dst)
            else:
                stem = p.stem
                suffix = p.suffix
                dst = self._unique_name(p.parent, f"{stem}_copy{suffix}")
                shutil.copy2(p, dst)
        except Exception as e:
            messagebox.showerror("Duplicate failed", f"{e}")
            return
        self._fb_rebuild_tree()

    def _unique_name(self, parent: Path, name: str) -> Path:
        candidate = parent / name
        if not candidate.exists():
            return candidate
        stem, suffix = os.path.splitext(name)
        n = 2
        while True:
            candidate = parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    def _fb_rename(self):
        p = self._fb_selected_path()
        if not p: return
        new_name = simpledialog.askstring("Rename", "New name:", initialvalue=p.name)
        if not new_name or new_name == p.name: return
        new_path = p.parent / new_name
        if new_path.exists():
            messagebox.showerror("Exists", "Target name already exists.")
            return
        try:
            p.rename(new_path)
        except Exception as e:
            messagebox.showerror("Rename failed", f"{e}")
            return
        self._fb_rebuild_tree()
        if self.filename and self.filename == p:
            self.filename = new_path
            self._update_title_status()

    def _fb_copy(self, mode="copy"):
        paths = []
        for iid in self.fb_tree.selection():
            fp = self.fb_tree.set(iid, "fullpath")
            if fp: paths.append(Path(fp))
        if not paths:
            p = self._fb_selected_path()
            if p: paths = [p]
        if not paths: return
        self.fs_clipboard = {"mode": mode, "paths": [str(x) for x in paths]}
        self.status.config(text=f"[{mode.UPPER()} set] {len(paths)} item(s)")

    def _fb_paste(self):
        if not self.fs_clipboard["paths"]:
            self.status.config(text="[Paste] Clipboard empty")
            return
        dest_dir = self._fb_selected_dir_for_new()
        if not dest_dir:
            messagebox.showinfo("Paste", "Select a destination folder first.")
            return
        mode = self.fs_clipboard["mode"]
        errors = []
        for spath in self.fs_clipboard["paths"]:
            sp = Path(spath)
            try:
                if mode == "copy":
                    if sp.is_dir():
                        dst = self._unique_name(dest_dir, sp.name)
                        shutil.copytree(sp, dst)
                    else:
                        dst = self._unique_name(dest_dir, sp.name)
                        shutil.copy2(sp, dst)
                elif mode == "move":
                    dst = dest_dir / sp.name
                    if dst.exists():
                        dst = self._unique_name(dest_dir, sp.name)
                    shutil.move(str(sp), str(dst))
            except Exception as e:
                errors.append(f"{sp.name}: {e}")
        self._fb_rebuild_tree()
        if errors:
            messagebox.showerror("Paste finished with errors", "\n".join(errors))
        else:
            self.status.config(text=f"[{mode.upper()}] Done")

    def _fb_delete(self):
        p = self._fb_selected_path()
        if not p: return
        if not messagebox.askyesno("Delete", f"Delete '{p.name}'?\nThis cannot be undone."):
            return
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            messagebox.showerror("Delete failed", f"{e}")
            return
        self._fb_rebuild_tree()

    # ---------- Pins ----------
    def _pins(self):
        return self.cfg.get("pins") or [None]*5

    def _save_pins(self, pins):
        self.cfg["pins"] = pins
        save_cfg(self.cfg)
        self._update_pin_button_labels()

    def _pin_button_click(self, idx: int, event):
        if event.state & 0x0001:  # Shift
            self._pin_set(idx)
        elif event.state & 0x0004:  # Control
            self._pin_clear(idx)
        else:
            self._pin_open(idx)

    def _pin_open(self, idx: int):
        pins = self._pins()
        p = pins[idx]
        if not p:
            self.status.config(text=f"[Pin {idx+1}] Empty")
            return
        path = Path(p)
        if not path.exists():
            messagebox.showwarning("Missing", f"Pinned file not found:\n{p}\n(Clearing this pin.)")
            pins[idx] = None
            self._save_pins(pins)
            return
        if not self._maybe_save_changes(): return
        try:
            data = path.read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = path
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)

    def _pin_set(self, idx: int):
        if not self.filename:
            messagebox.showinfo("Pin", "Save/open a file first.")
            return
        pins = self._pins()
        pins[idx] = str(self.filename)
        self._save_pins(pins)
        self.status.config(text=f"[Pin {idx+1}] Set to {self.filename.name}")

    def _pin_clear(self, idx: int):
        pins = self._pins()
        pins[idx] = None
        self._save_pins(pins)
        self.status.config(text=f"[Pin {idx+1}] Cleared")

    def _update_pin_button_labels(self):
        pins = self._pins()
        for i, btn in enumerate(self.pin_buttons):
            btn.configure(text=f"Pin{i+1}")

    # ---------- Bind keys ----------
    def _bind_keys(self):
        self.bind_all("<Control-s>", lambda e: (self.save_file(), "break"))
        self.bind_all("<Control-o>", lambda e: (self.open_file(), "break"))
        self.bind_all("<Control-n>", lambda e: (self.new_file(), "break"))
        self.bind_all("<Control-a>", self._select_all)
        self.bind_all("<Control-z>", lambda e: (self.text.event_generate("<<Undo>>"), "break"))
        self.bind_all("<Control-y>", lambda e: (self.text.event_generate("<<Redo>>"), "break"))
        self.bind_all("<F5>", lambda e: (self.run_current(), "break"))
        self.bind_all("<Control-Shift-D>", lambda e: (self.toggle_dark_mode(), "break"))
        self.bind_all("<Control-w>", lambda e: (self._close_active_tab(), "break"))

    # ---------- Theme ----------
    def toggle_dark_mode(self):
        self.cfg["dark_mode"] = not self.cfg.get("dark_mode", True)
        save_cfg(self.cfg)
        self.theme = DARK if self.cfg["dark_mode"] else LIGHT
        self._apply_theme()
        self._schedule_line_numbers()

    def _apply_theme(self):
        t = self.theme
        # Root areas
        self.configure(bg=t["bg"])
        self.toolbar_outer.configure(bg=t["toolbar_bg"])
        self.status_outer.configure(bg=t["status_bg"])
        self.status.configure(background=t["status_bg"], foreground=t["status_fg"])

        # Text + gutter + search highlight
        self.text.configure(bg=t["bg"], fg=t["fg"], insertbackground=t["cursor"],
                            selectbackground=t["sel_bg"], selectforeground=t["sel_fg"])
        self.text.tag_configure("search_hit", background=t["search_bg"], foreground=t["search_fg"])
        self.gutter_canvas.configure(bg=t["gutter_bg"])

        # Menus
        self.option_add("*Menu.background", t["menu_bg"])
        self.option_add("*Menu.foreground", t["menu_fg"])
        self.option_add("*Menu.activeBackground", t["menu_active"])
        self.option_add("*Menu.activeForeground", t["menu_fg"])

        # ttk styling
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background=t["bg"])
        style.configure("Tool.TFrame", background=t["toolbar_bg"])
        self.toolbar.configure(style="Tool.TFrame")

        style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        style.configure("TButton", background=t["toolbar_bg"], foreground=t["fg"], padding=6)
        style.map("TButton",
                  background=[("active", t["tab_sel_bg"])],
                  foreground=[("active", t["tab_sel_fg"])])

        style.configure("TEntry", fieldbackground=t["bg"], foreground=t["fg"])
        style.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        style.configure("Vertical.TScrollbar", background=t["bg"])
        style.configure("Horizontal.TScrollbar", background=t["bg"])

        style.configure("TNotebook", background=t["bg"])
        style.configure("TNotebook.Tab", background=t["tab_bg"], foreground=t["tab_fg"], padding=(10, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", t["tab_sel_bg"])],
                  foreground=[("selected", t["tab_sel_fg"])])

        style.configure('Treeview',
                        background=t['bg'], foreground=t['fg'],
                        fieldbackground=t['bg'], borderwidth=0)
        style.map('Treeview',
                  background=[('selected', t['tree_sel_bg'])],
                  foreground=[('selected', t['tree_sel_fg'])])
        style.configure('Treeview.Heading',
                        background=t['toolbar_bg'], foreground=t['fg'])

    # ---------- File ops ----------
    def new_file(self):
        if not self._maybe_save_changes(): return
        self.filename = None
        self.text.delete("1.0", tk.END)
        self._modified = False
        self._update_title_status()
        self._schedule_line_numbers()

    def open_file(self):
        if not self._maybe_save_changes(): return
        path = filedialog.askopenfilename(title="Open file")
        if not path: return
        try:
            data = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Open failed", f"{e}")
            return
        self.filename = Path(path)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", data)
        self._modified = False
        self._update_title_status()
        self.project_root = self._find_pkg_root(self.filename.parent)
        if not self.cfg.get("fb_root"):
            self.fb_root = self.project_root or self.fb_root
            self._fb_rebuild_tree()
        self._schedule_line_numbers()

    def save_file(self):
        if self.filename is None:
            return self.save_file_as()
        try:
            data = self.text.get("1.0", tk.END)
            self.filename.write_text(data.rstrip("\n") + "\n", encoding="utf-8")
            self._modified = False
            self._update_title_status()
            self.status.config(text=f"Saved to {self.filename}")
            self.project_root = self._find_pkg_root(self.filename.parent)
            return True
        except Exception as e:
            messagebox.showerror("Save failed", f"{e}")
            return False

    def save_file_as(self):
        path = filedialog.asksaveasfilename(title="Save As", defaultextension="",
                                            initialfile=(self.filename.name if self.filename else "untitled.txt"))
        if not path: return False
        self.filename = Path(path)
        self.project_root = self._find_pkg_root(self.filename.parent)
        return self.save_file()

    def restart_app(self):
        if not self._maybe_save_changes():
            return
        try:
            for c in self.consoles:
                try:
                    if c["proc"]:
                        c["proc"].terminate()
                except Exception:
                    pass
        except Exception:
            pass
        python = sys.executable
        args = [python] + sys.argv
        self.destroy()
        os.execv(python, args)

    # ---------- Edit ops ----------
    def _select_all(self, event=None):
        self.text.tag_add(tk.SEL, "1.0", tk.END)
        self.text.mark_set(tk.INSERT, "1.0")
        self.text.see(tk.INSERT)
        return "break"

    def copy_all(self):
        data = self.text.get("1.0", tk.END).rstrip()
        self.clipboard_clear()
        self.clipboard_append(data)
        self.status.config(text="[All code copied]")
        self.after(1500, self._update_title_status)

    # ---------- Run integration ----------
    def run_current(self):
        if self.filename is None:
            choice = messagebox.askyesnocancel(
                "Save file?",
                "This buffer has never been saved.\n"
                "Yes = Save As and run\n"
                "No = Run a temp copy\n"
                "Cancel = Abort"
            )
            if choice is None:
                return
            if choice:
                if not self.save_file_as(): return
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as tmpf:
                    tmpf.write(self.text.get("1.0", tk.END))
                    tmp_path = Path(tmpf.name)
                self._start_process_for_path(tmp_path, temp_mode=True)
                return
        else:
            if self._modified and not self.save_file():
                return
        self._start_process_for_path(self.filename, temp_mode=False)

    def _start_process_for_path(self, path: Path, temp_mode: bool):
        cmd = self._compute_run_cmd_for(path)
        con = self._current_console()
        if not con:
            self.new_console_tab()
            con = self._current_console()
        self._console_clear(con)
        self._console_write(con, f"$ {cmd}\n")
        try:
            con["proc"] = subprocess.Popen(
                cmd, shell=True, cwd=str(path.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True, bufsize=1,
                creationflags=CREATE_FLAGS
            )
            self.btn_run.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)
            threading.Thread(target=self._pump_output, args=(con,), daemon=True).start()
        except Exception as e:
            self._console_write(con, f"[error] {e}\n")
        if temp_mode:
            def cleanup():
                if con["proc"] is None:
                    try: path.unlink(missing_ok=True)
                    except Exception: pass
                else:
                    self.after(500, cleanup)
            self.after(500, cleanup)

    def stop_run(self):
        con = self._current_console()
        if not con: return
        p = con.get("proc")
        if not p: return
        try:
            p.terminate()
            self._console_write(con, "\n[process terminated]\n")
        except Exception as e:
            self._console_write(con, f"\n[stop error] {e}\n")
        finally:
            con["proc"] = None
            self.btn_run.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)

    def _pump_output(self, con):
        p = con["proc"]
        try:
            for line in p.stdout:
                self._console_write(con, line)
        except Exception as e:
            self._console_write(con, f"\n[stream error] {e}\n")
        finally:
            rc = p.wait()
            self._console_write(con, f"\n[process exited with code {rc}]\n")
            con["proc"] = None
            self.after(0, lambda: (self.btn_run.configure(state=tk.NORMAL), self.btn_stop.configure(state=tk.DISABLED)))

    def _compute_run_cmd_for(self, path: Path) -> str:
        # Always prefer PowerShell; no raw cmd.exe
        ext = path.suffix.lower()
        if ext == ".ps1": return f'powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "{path}"'
        if ext == ".py":  return f'powershell -NoLogo -NoProfile -Command python "{path}"'
        if ext == ".js":  return f'powershell -NoLogo -NoProfile -Command node "{path}"'
        if ext in (".bat", ".cmd"): return f'powershell -NoLogo -NoProfile -Command "& \'{path}\'"'
        if os.access(path, os.X_OK): return f'powershell -NoLogo -NoProfile -Command "& \'{path}\'"'
        return f'powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "{path}"'

    # ---------- Console tabs (PowerShell) ----------
    def new_console_tab(self):
        idx_plus = self.console_tabs.index(self.plus_tab)
        frame = ttk.Frame(self.console_tabs)

        # Top: output
        top = ttk.Frame(frame)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        text = tk.Text(top, height=12, wrap="word", state=tk.DISABLED, font=MONO_FONT)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(top, orient="vertical", command=text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll.set)

        # Bottom: dual input bars
        bottom = ttk.Frame(frame, padding=(0,6))
        bottom.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bottom, text="Command:").grid(row=0, column=0, padx=(0,6))
        cmd_entry = ttk.Entry(bottom)
        cmd_entry.grid(row=0, column=1, sticky="ew", padx=(0,6))
        cmd_btn = ttk.Button(bottom, text="Send", command=lambda: self._send_command(console))
        cmd_btn.grid(row=0, column=2)

        ttk.Label(bottom, text="Prompt:").grid(row=1, column=0, padx=(0,6), pady=(6,0))
        prompt_entry = ttk.Entry(bottom)
        prompt_entry.grid(row=1, column=1, sticky="ew", padx=(0,6), pady=(6,0))
        prompt_btn = ttk.Button(bottom, text="Send", command=lambda: self._send_prompt(console))
        prompt_btn.grid(row=1, column=2, pady=(6,0))

        bottom.columnconfigure(1, weight=1)

        # Launch PowerShell (hidden window)
        try:
            proc = subprocess.Popen(
                "powershell -NoLogo -NoProfile",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=CREATE_FLAGS
            )
        except Exception as e:
            proc = None
            text.configure(state=tk.NORMAL)
            text.insert(tk.END, f"[Error launching PowerShell: {e}]\n")
            text.configure(state=tk.DISABLED)

        console = {"frame": frame, "text": text, "cmd_entry": cmd_entry, "prompt_entry": prompt_entry, "proc": proc}
        self.consoles.append(console)

        # Events
        cmd_entry.bind("<Return>", lambda e: (self._send_command(console), "break"))
        prompt_entry.bind("<Return>", lambda e: (self._send_prompt(console), "break"))

        # Theme on text area
        t = self.theme
        text.configure(bg=t["console_bg"], fg=t["console_fg"], insertbackground=t["cursor"])

        # Insert before "+" and re-add "+" to end
        title = "× PowerShell"
        self.console_tabs.insert(idx_plus, frame, text=title)
        self.console_tabs.forget(self.plus_tab)
        self.console_tabs.add(self.plus_tab, text="+")
        self.console_tabs.select(frame)

        if proc:
            threading.Thread(target=self._pump_console_output, args=(console,), daemon=True).start()

    def _current_console(self):
        try:
            sel = self.console_tabs.select()
            for c in self.consoles:
                if str(c["frame"]) == sel:
                    return c
        except Exception:
            return None
        return None

    def _pump_console_output(self, console):
        p = console["proc"]
        try:
            for line in p.stdout:
                self._console_write(console, line)
        except Exception as e:
            self._console_write(console, f"\n[stream error] {e}\n")
        finally:
            try:
                rc = p.wait()
            except Exception:
                rc = "?"
            self._console_write(console, f"\n[console exited with code {rc}]\n")

    def _send_command(self, console):
        data = console["cmd_entry"].get().strip()
        console["cmd_entry"].delete(0, tk.END)
        if not data: return
        self._console_write(console, f"> {data}\n")
        p = console.get("proc")
        if not p or not p.stdin:
            self._console_write(console, "[no console process]\n")
            return
        try:
            p.stdin.write(data + "\n")
            p.stdin.flush()
        except Exception as ex:
            self._console_write(console, f"[input error] {ex}\n")

    def _send_prompt(self, console):
        data = console["prompt_entry"].get()
        console["prompt_entry"].delete(0, tk.END)
        if data == "": return
        self._console_write(console, data + "\n")

    def _close_console_tab(self, console):
        try:
            if console.get("proc"):
                try:
                    console["proc"].terminate()
                except Exception:
                    pass
        finally:
            try:
                self.console_tabs.forget(console["frame"])
            except Exception:
                pass
            self.consoles = [c for c in self.consoles if c is not console]

    def _console_clear(self, console):
        t = console["text"]
        t.configure(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        t.configure(state=tk.DISABLED)

    def _console_write(self, console, text: str):
        t = console["text"]
        t.configure(state=tk.NORMAL)
        t.insert(tk.END, text)
        t.see(tk.END)
        t.configure(state=tk.DISABLED)

    # ---------- Scrolling / line numbers ----------
    def _yview_text(self, *args):
        self.text.yview(*args)
        self._schedule_line_numbers()

    def _yscrollcommand(self, sb, first, last):
        sb.set(first, last)
        self._schedule_line_numbers()

    def _schedule_line_numbers(self):
        # Throttle: draw once per idle cycle
        if getattr(self, "_ln_sched", False): return
        self._ln_sched = True
        self.after_idle(self._update_line_numbers)

    def _update_line_numbers(self):
        # Guard against early callbacks during teardown/early init
        if not self.winfo_exists() or not self.text.winfo_exists() or not self.gutter_canvas.winfo_exists():
            return
        self._ln_sched = False
        t = self.theme
        canvas = self.gutter_canvas
        canvas.delete("all")

        total_lines = int(self.text.index("end-1c").split(".")[0])
        digits = max(3, len(str(total_lines)))
        char_w = self._ln_font.measure("9")
        pad = 10
        width_px = digits * char_w + pad
        canvas.configure(width=width_px, bg=t["gutter_bg"])

        index = self.text.index("@0,0")
        height = self.text.winfo_height()
        fg = t["gutter_fg"]

        while True:
            dli = self.text.dlineinfo(index)
            if dli is None:
                break
            y = dli[1]
            if y > height:
                break
            line_no = int(index.split(".")[0])
            canvas.create_text(width_px - 6, y, anchor="ne", text=str(line_no), fill=fg, font=self._ln_font)
            index = self.text.index(f"{line_no + 1}.0")

    # ---------- Helpers ----------
    def _update_title_status(self):
        name = (self.filename.name if self.filename else "untitled")
        mod = " *" if self._modified else ""
        root_hint = f"  {self.project_root}" if self.project_root else ""
        self.title(f"{APP_TITLE} - {name}{mod}")
        idx = self.text.index(tk.INSERT)
        line, col = idx.split(".")
        self.status.config(text=f"{name}{mod}{root_hint}   |   Line {line}, Col {int(col)+1}")

    def _on_modified(self, event=None):
        if self.text.edit_modified():
            self._modified = True
            self.text.edit_modified(False)
        self._update_title_status()
        self._schedule_line_numbers()

    def _on_close(self):
        if not self._maybe_save_changes(): return
        try:
            for c in self.consoles:
                try:
                    if c["proc"]:
                        c["proc"].terminate()
                except Exception:
                    pass
        finally:
            self.destroy()

    def _maybe_save_changes(self) -> bool:
        if not self._modified:
            return True
        resp = messagebox.askyesnocancel("Unsaved changes", "Save changes before closing?")
        if resp is None: return False
        if resp: return bool(self.save_file())
        return True

    # ---------- Project helpers ----------
    def _find_pkg_root(self, start_dir: Path | None) -> Path | None:
        if not start_dir:
            return None
        d = start_dir.resolve()
        for _ in range(30):
            if (d / "package.json").exists():
                return d
            if d.parent == d:
                break
            d = d.parent
        return None

    def _set_project_root(self):
        d = filedialog.askdirectory(title="Choose Project Root")
        if not d:
            return
        self.project_root = Path(d)
        self.status.config(text=f"Project root set: {self.project_root}")

if __name__ == "__main__":
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = MiniNano()
    app.mainloop()

