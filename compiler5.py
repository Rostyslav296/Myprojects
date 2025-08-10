# Clang-compiler (A customized fork of Clang compiler from the LLVM project.)

import subprocess
import os
import sys
import zipfile
import urllib.request
import shutil
import stat
from pathlib import Path
import platform
import glob
import tkinter as tk
from tkinter import filedialog
import tarfile
import re
import time

# Suppress Tkinter deprecation warning
os.environ["TK_SILENCE_DEPRECATION"] = "1"

# Helper functions from base script
def display_menu():
    print("\n=== Clang & N64 Compiler Menu ===")
    print("1. Compile C file (host formats)")
    print("2. Setup N64 Toolchain (required for .z64)")
    print("3. Exit")

def get_file_path():
    current_dir = os.getcwd()
    c_files = glob.glob(os.path.join(current_dir, '*.c'))
    if c_files:
        print("\nAvailable .c files in current directory:")
        for i, f in enumerate([os.path.basename(f) for f in c_files], 1):
            print(f"{i}. {f}")

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        initialdir=current_dir,
        title="Select C file",
        filetypes=(("C files", "*.c"), ("All files", "*.*"))
    )
    root.destroy()

    if file_path and os.path.exists(file_path) and file_path.endswith('.c'):
        return file_path
    else:
        print("No valid .c file selected. Falling back to manual input.")
        while True:
            input_val = input("\nEnter the path to your .c file or select number (e.g., 1): ").strip()
            if input_val.isdigit():
                num = int(input_val)
                if 1 <= num <= len(c_files):
                    return c_files[num-1]
                else:
                    print("Invalid number. Please select a valid number or enter a file path.")
            else:
                if os.path.exists(input_val) and input_val.endswith('.c'):
                    return input_val
                print("Invalid file path or not a .c file. Try again.")

def choose_format():
    print("\nChoose output format:")
    print("1. Executable (binary)")
    print("2. Object file (.o)")
    print("3. Assembly code (.s)")
    print("4. LLVM IR (.ll)")
    print("5. N64 ROM (.z64)")
    while True:
        choice = input("Select (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice. Please select a number between 1 and 5.")

def check_for_libdragon(file_path):
    """Check if the file includes libdragon.h."""
    try:
        with open(file_path, 'r') as f:
            return any('libdragon.h' in line for line in f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def compile_host(file_path, format_choice):
    base_name = os.path.splitext(file_path)[0]
    is_n64_code = check_for_libdragon(file_path)

    if is_n64_code and format_choice != '5':
        if not is_built("gcc-stage1"):
            print("Error: File includes libdragon.h. N64 Toolchain is required for compilation.")
            print("Please run option 2 to set up the N64 Toolchain first.")
            return
        cmd = ['mips64-elf-gcc', f'-I{N64_INST}/include']
    else:
        cmd = ['clang']

    if format_choice == '1':
        cmd += [file_path, '-o', base_name]
    elif format_choice == '2':
        cmd += ['-c', file_path, '-o', base_name + '.o']
    elif format_choice == '3':
        cmd += ['-S', file_path, '-o', base_name + '.s']
    elif format_choice == '4':
        cmd += ['-S', '-emit-llvm', file_path, '-o', base_name + '.ll']
    else:
        print("Invalid choice.")
        return

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\nCompilation successful!")
        print("Output file:", base_name + ('' if format_choice == '1' else {'2': '.o', '3': '.s', '4': '.ll'}[format_choice]))
        if result.stdout:
            print("Output:\n", result.stdout)
        if result.stderr:
            print("Warnings/Errors:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Compilation failed.")
        print("Error:", e.stderr)

# N64 Toolchain Setup (Barebones, adapted for minimal)
WORK_DIR = Path.home() / ".n64dev"
N64SDK_DIR = WORK_DIR / "n64sdk"
N64_INST = N64SDK_DIR / "install"
os.environ["N64_INST"] = str(N64_INST)
os.environ["PATH"] = f"{N64_INST / 'bin'}:{os.environ.get('PATH', '')}"

ZIP_URLS = {
    "gmp-6.3.0.tar.xz": "https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz",
    "mpfr-4.2.2.tar.xz": "https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.2.tar.xz",
    "mpc-1.3.1.tar.gz": "https://ftp.gnu.org/gnu/mpc/mpc-1.3.1.tar.gz",
    "binutils-2.45.tar.gz": "https://ftp.gnu.org/gnu/binutils/binutils-2.45.tar.gz",
    "gcc-15.1.0.tar.gz": "https://ftp.gnu.org/gnu/gcc/gcc-15.1.0/gcc-15.1.0.tar.gz",
    "newlib-4.4.0.20231231.tar.gz": "https://sourceware.org/pub/newlib/newlib-4.4.0.20231231.tar.gz",
}

LIBDRAGON_REPO = "https://github.com/DragonMinded/libdragon.git"

def ensure_write_permissions(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if str(path).startswith(str(WORK_DIR)) or path == WORK_DIR:
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        except PermissionError:
            print(f"Error: Cannot set permissions for {path}. Please run as sudo or fix permissions manually.")
            sys.exit(1)

def robust_download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    ensure_write_permissions(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    
    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")
    
    fallback_urls = {
        "newlib-4.4.0.20231231.tar.gz": [
            "https://sourceware.mirror.garr.it/newlib/newlib-4.4.0.20231231.tar.gz",
            "https://ftp.redhat.com/redhat/newlib/newlib-4.4.0.20231231.tar.gz",
        ]
    }
    
    def try_download(url_to_try, dest_file):
        nonlocal downloaded
        req = urllib.request.Request(url_to_try)
        if downloaded > 0:
            req.add_header("Range", f"bytes={downloaded}-")
        try:
            with urllib.request.urlopen(req) as resp:
                total = downloaded + int(resp.headers.get('Content-Length', 0))
                with dest_file.open("ab") as f:
                    chunk_size = 1024 * 8
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            percent = downloaded / total * 100
                            sys.stdout.write(f"\rDownloading {dest.name}: {percent:.2f}%")
                            sys.stdout.flush()
                print()  # New line after completion
            return True
        except urllib.error.HTTPError as e:
            print(f"Download failed for {url_to_try}: {e}")
            return False

    try:
        if try_download(url, tmp):
            tmp.replace(dest)
            return
    except urllib.error.URLError as e:
        print(f"Download failed for {url}: {e}")

    # Try fallback URLs if available
    if dest.name in fallback_urls:
        for fallback_url in fallback_urls[dest.name]:
            print(f"Trying fallback URL: {fallback_url}")
            if try_download(fallback_url, tmp):
                tmp.replace(dest)
                return
        print(f"All download attempts failed for {dest.name}. Please check the URLs or download the file manually.")
        print("Visit https://sourceware.org/newlib/ for the latest newlib releases.")
        sys.exit(1)
    else:
        raise

def extract_archive(archive_path, dest_dir):
    dest_dir = Path(dest_dir)
    ensure_write_permissions(dest_dir)
    
    # Remove existing directory to avoid conflicts
    extract_dir = dest_dir / archive_path.stem.replace('.tar', '')
    if extract_dir.exists():
        print(f"Removing existing directory {extract_dir} to avoid conflicts")
        shutil.rmtree(extract_dir, ignore_errors=True)
    
    try:
        if archive_path.suffix in ['.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:gz') as tar:
                members = tar.getmembers()
                total_members = len(members)
                for i, member in enumerate(members):
                    tar.extract(member, dest_dir)
                    percent = (i + 1) / total_members * 100
                    sys.stdout.write(f"\rExtracting {archive_path.name}: {percent:.2f}%")
                    sys.stdout.flush()
                print()
        elif archive_path.suffix in ['.xz']:
            with tarfile.open(archive_path, 'r:xz') as tar:
                members = tar.getmembers()
                total_members = len(members)
                for i, member in enumerate(members):
                    tar.extract(member, dest_dir)
                    percent = (i + 1) / total_members * 100
                    sys.stdout.write(f"\rExtracting {archive_path.name}: {percent:.2f}%")
                    sys.stdout.flush()
                print()
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path) as z:
                infos = z.infolist()
                total_infos = len(infos)
                for i, info in enumerate(infos):
                    z.extract(info, dest_dir)
                    percent = (i + 1) / total_infos * 100
                    sys.stdout.write(f"\rExtracting {archive_path.name}: {percent:.2f}%")
                    sys.stdout.flush()
                print()
    except PermissionError as e:
        print(f"Permission error during extraction: {e}")
        print(f"Please ensure you have write permissions for {dest_dir} or run as sudo.")
        sys.exit(1)
    finally:
        archive_path.unlink(missing_ok=True)

def build_env():
    env = os.environ.copy()
    if platform.system() == "Darwin":
        env.update(
            CC="clang",
            CXX="clang++",
            CFLAGS="-Wno-error -Wno-implicit-function-declaration",
            CXXFLAGS="-Wno-error",
            CPPFLAGS=f"-I{N64_INST}/include",
            LDFLAGS=f"-L{N64_INST}/lib",
        )
    return env

def is_built(name):
    """Check if a component is already built."""
    if name.startswith("gcc"):
        return (N64_INST / "bin" / "mips64-elf-gcc").exists()
    elif name == "binutils-2.45":
        return (N64_INST / "bin" / "mips64-elf-ld").exists()
    elif name == "newlib-4.4.0":
        return (N64_INST / "mips64-elf" / "lib" / "libc.a").exists()
    elif name == "libdragon-toolchain-continuous-prerelease":
        return (N64SDK_DIR / "libdragon" / "lib" / "libdragon.a").exists()
    return (N64_INST / "lib" / f"lib{name.split('-')[0]}.a").exists() or (N64_INST / "lib64" / f"lib{name.split('-')[0]}.a").exists()

def count_source_files(build_dir):
    """Estimate the number of source files to compile for progress estimation."""
    c_files = len(list(build_dir.rglob("*.c")))
    cpp_files = len(list(build_dir.rglob("*.cpp")))
    return c_files + cpp_files

def run_make_with_progress(cmd, build_dir, stage_name, component_name):
    """Run make command with a progress bar based on compiled files or output lines."""
    total_files = count_source_files(build_dir) or 200  # Increased fallback for larger projects
    compiled_files = 0
    max_lines = 2000  # Increased for larger builds like gcc
    line_count = 0
    start_time = time.time()
    last_update = start_time
    
    process = subprocess.Popen(cmd, env=build_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while process.poll() is None:
        line = process.stdout.readline()
        if line:
            print(line, end='')  # Add this line to print output in real-time
            line_count += 1
            if re.search(r'\b(gcc|cc|clang)\b.*\.(c|cpp)\b', line):
                compiled_files += 1
            # Use file-based progress for make, line-based for scripts like build.sh
            if 'build.sh' in cmd:
                percent = min(line_count / max_lines * 100, 100)
            else:
                percent = min(compiled_files / total_files * 100, 100)
            # Update progress only every 0.1 seconds to reduce flicker
            if time.time() - last_update > 0.1:
                sys.stdout.write(f"\r{stage_name} {component_name}: {percent:.2f}%")
                sys.stdout.flush()
                last_update = time.time()
    print()
    
    if process.returncode != 0:
        print(f"{stage_name} failed for {component_name}.")
        sys.exit(1)

def setup_n64_toolchain():
    WORK_DIR.mkdir(exist_ok=True)
    ensure_write_permissions(WORK_DIR)
    os.chdir(WORK_DIR)
    env = build_env()

    N64SDK_DIR.mkdir(exist_ok=True)

    used_brew = False
    if platform.system() == "Darwin":
        print("[Using Homebrew for macOS to install mips64-elf toolchain]")
        try:
            # Unset common git config that causes invalid username errors
            subprocess.call(["git", "config", "--global", "--unset", "url.git@github.com:.insteadOf"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.call(["git", "config", "--global", "--unset", "url.ssh://git@github.com/.insteadOf"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Manual clone to bypass potential git config/clone issues
            tap_dir = "/opt/homebrew/Library/Taps/messense/homebrew-macos-cross-toolchains"
            if not os.path.exists(tap_dir):
                os.makedirs(os.path.dirname(tap_dir), exist_ok=True)
                subprocess.check_call(["git", "clone", "https://github.com/messense/homebrew-macos-cross-toolchains.git", tap_dir])
            else:
                # If dir exists, pull updates
                os.chdir(tap_dir)
                subprocess.check_call(["git", "pull"])
                os.chdir(WORK_DIR)

            # Tap explicitly with URL (redundant after manual clone but harmless)
            subprocess.check_call(["brew", "tap", "messense/macos-cross-toolchains", "https://github.com/messense/homebrew-macos-cross-toolchains.git"])

            # Install the correct formula
            subprocess.check_call(["brew", "install", "mips64-unknown-elf-gcc"])

            # Set up symlinks in N64_INST for consistency (rename by removing '-unknown')
            N64_INST.mkdir(parents=True, exist_ok=True)
            bin_dir = N64_INST / "bin"
            bin_dir.mkdir(exist_ok=True)
            brew_bin = "/opt/homebrew/bin" if platform.machine() == "arm64" else "/usr/local/bin"
            for tool in glob.glob(f"{brew_bin}/mips64-unknown-elf-*"):
                tool_name = os.path.basename(tool).replace('-unknown', '')
                target = bin_dir / tool_name
                if target.exists():
                    target.unlink()
                target.symlink_to(tool)
            used_brew = True
            print("[Skipping source build for binutils, gcc, newlib; using brew versions]")
        except subprocess.CalledProcessError as e:
            print(f"Brew install failed: {e}. Falling back to source build with fixes.")
            print("If git clone fails with 'invalid username', run: git config --global --unset url.git@github.com:.insteadOf")
            used_brew = False

    if not used_brew:
        # Download and extract sources
        for filename, url in ZIP_URLS.items():
            zip_path = N64SDK_DIR / filename
            if not zip_path.exists():
                print(f"[download] {filename}")
                robust_download(url, zip_path)
            print(f"[extract] {filename}")
            extract_archive(zip_path, N64SDK_DIR)

        # Patch libgloss/mips/test.c to fix GCC 14.x build issues
        newlib_dir = N64SDK_DIR / "newlib-4.4.0.20231231"
        test_c_path = newlib_dir / "libgloss" / "mips" / "test.c"
        if test_c_path.exists():
            with open(test_c_path, 'r') as f:
                content = f.read()
            # Add declarations for implicit functions and fix return type/value
            patch = """
extern void outbyte(char c);
extern void print(const char *ptr);
"""
            content = patch + content.replace('main()', 'int main(void)').replace('return;', 'return 0;')
            with open(test_c_path, 'w') as f:
                f.write(content)
            print("[Applied patch to libgloss/mips/test.c for GCC compatibility]")

        # Patch libgloss/mips/syscalls.c to fix type mismatches and implicit declarations
        syscalls_c_path = newlib_dir / "libgloss" / "mips" / "syscalls.c"
        if syscalls_c_path.exists():
            with open(syscalls_c_path, 'r') as f:
                content = f.read()
            # Add struct and extern declaration
            patch = """
struct _memory_region_ {
    char *address;
    size_t size;
};
extern void get_mem_info(struct _memory_region_ *region);
"""
            content = content.replace('#include <sys/stat.h>', '#include <sys/stat.h>\n' + patch)
            # Fix comparisons and assignment
            content = content.replace('((size_t)heap_ptr >= heap_start)', '((size_t)heap_ptr >= (size_t)heap_start)')
            content = content.replace('((size_t)heap_ptr < (heap_start + mem.size))', '((size_t)heap_ptr < (size_t)(heap_start + mem.size))')
            content = content.replace('avail = (heap_start + mem.size) - (size_t)heap_ptr;', 'avail = (size_t)(heap_start + mem.size) - (size_t)heap_ptr;')
            with open(syscalls_c_path, 'w') as f:
                f.write(content)
            print("[Applied patch to libgloss/mips/syscalls.c for build compatibility]")

        # Build steps (barebones: gmp, mpfr, mpc, binutils, gcc stage1, newlib, gcc stage2)
        steps = [
            {"name": "gmp-6.3.0", "dir": N64SDK_DIR / "gmp-6.3.0", "cfg": ["../configure", f"--prefix={N64_INST}"]},
            {"name": "mpfr-4.2.2", "dir": N64SDK_DIR / "mpfr-4.2.2", "cfg": ["../configure", f"--prefix={N64_INST}", f"--with-gmp={N64_INST}"]},
            {"name": "mpc-1.3.1", "dir": N64SDK_DIR / "mpc-1.3.1", "cfg": ["../configure", f"--prefix={N64_INST}", f"--with-gmp={N64_INST}", f"--with-mpfr={N64_INST}", "--disable-shared"]},
            {"name": "binutils-2.45", "dir": N64SDK_DIR / "binutils-2.45", "cfg": ["../configure", f"--prefix={N64_INST}", "--target=mips64-elf", "--disable-werror"]},
            {"name": "gcc-stage1", "dir": N64SDK_DIR / "gcc-15.1.0", "cfg": ["../configure", f"--prefix={N64_INST}", "--target=mips64-elf", "--enable-languages=c", "--disable-multilib", "--without-headers", "--disable-shared", "--disable-threads", "--disable-werror", "--with-abi=64", "--with-system-zlib", f"--with-gmp={N64_INST}", f"--with-mpfr={N64_INST}", f"--with-mpc={N64_INST}"], "build_target": "all-gcc", "install_target": "install-gcc"},
            {"name": "newlib-4.4.0", "dir": N64SDK_DIR / "newlib-4.4.0.20231231", "cfg": ["../configure", f"--prefix={N64_INST}", "--target=mips64-elf", f"--with-sysroot={N64_INST}/mips64-elf", "--disable-newlib-io-c99-formats", "--disable-newlib-io-long-long", "--disable-newlib-io-float", "--disable-newlib-io-long-double", "--disable-newlib-io-complex", "--disable-newlib-supplied-syscalls", "--enable-newlib-reent-small", "--disable-newlib-fvwrite-in-streamio", "--disable-newlib-fseek-optimization", "--disable-newlib-nano-formatted-io", "--disable-newlib-unbuf-stream-opt"]},
            {"name": "gcc-stage2", "dir": N64SDK_DIR / "gcc-15.1.0", "cfg": ["../configure", f"--prefix={N64_INST}", "--target=mips64-elf", "--enable-languages=c,c++", "--disable-multilib", "--with-newlib", "--disable-werror", "--with-abi=64", "--with-system-zlib", f"--with-gmp={N64_INST}", f"--with-mpfr={N64_INST}", f"--with-mpc={N64_INST}"]},
        ]

        if platform.system() == "Darwin":
            sysroot = "--with-sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
            for step in steps:
                if step["name"].startswith("gcc-"):
                    step["cfg"].append(sysroot)

        for step in steps:
            if is_built(step["name"]):
                print(f"[skip] {step['name']}")
                continue
            build_dir = step["dir"] / "build"
            build_dir.mkdir(exist_ok=True)
            ensure_write_permissions(build_dir)
            os.chdir(build_dir)
            print(f"[configure] {step['name']}")
            configure_script = step["dir"] / "configure"
            if not configure_script.exists():
                print(f"Error: Configure script not found at {configure_script}. Ensure the source directory {step['dir']} is correctly extracted.")
                sys.exit(1)
            if platform.system() == "Darwin" and step["name"] == "binutils-2.45":
                cache_file = "config.cache"
                with open(cache_file, "w") as f:
                    f.write("ac_cv_func_strncmp=yes\n")
                    f.write("ac_cv_func_strncmp_works=yes\n")
                step["cfg"].append(f"--cache-file={cache_file}")
            if step["name"] == "newlib-4.4.0":
                cache_file = "config.cache"
                with open(cache_file, "w") as f:
                    f.write("ac_cv_objext=o\n")
                    f.write("ac_cv_type_long_double=no\n")
                step["cfg"].append(f"--cache-file={cache_file}")
                env['CFLAGS_FOR_TARGET'] = '-Wno-error=implicit-function-declaration -Wno-implicit-int'
            subprocess.check_call(step["cfg"], env=env)
            print(f"[build] {step['name']}")
            make_cmd = ["make"]
            if "build_target" in step:
                make_cmd.append(step["build_target"])
            run_make_with_progress(make_cmd, step["dir"], "Building", step["name"])
            print(f"[install] {step['name']}")
            install_cmd = ["make", "install"]
            if "install_target" in step:
                install_cmd = ["make", step["install_target"]]
            run_make_with_progress(install_cmd, step["dir"], "Installing", step["name"])
            if step["name"] == "gcc-stage1":
                # Copy fixed headers to sysroot for newlib
                sysroot_dir = N64_INST / "mips64-elf" / "include"
                sysroot_dir.mkdir(parents=True, exist_ok=True)
                fixed_dir = N64_INST / "lib" / "gcc" / "mips64-elf" / "15.1.0" / "include"
                fixed_fixed_dir = N64_INST / "lib" / "gcc" / "mips64-elf" / "15.1.0" / "include-fixed"
                if fixed_dir.exists():
                    shutil.copytree(fixed_dir, sysroot_dir, dirs_exist_ok=True)
                if fixed_fixed_dir.exists():
                    shutil.copytree(fixed_fixed_dir, sysroot_dir, dirs_exist_ok=True)

    # Build libdragon
    libdragon_dir = N64SDK_DIR / "libdragon"
    if not libdragon_dir.exists():
        print("[clone] libdragon")
        subprocess.check_call(["git", "clone", LIBDRAGON_REPO, str(libdragon_dir)])
    os.chdir(libdragon_dir)
    if is_built("libdragon-toolchain-continuous-prerelease"):
        print("[skip] libdragon")
    else:
        print("[build] libdragon")
        run_make_with_progress(["./build.sh"], libdragon_dir, "Building", "libdragon")
    print("[N64 Toolchain Setup Complete]")

def compile_n64(file_path):
    if not is_built("libdragon-toolchain-continuous-prerelease"):
        print("N64 Toolchain not set up. Running setup...")
        setup_n64_toolchain()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_dir = Path("/tmp/n64_build")
    temp_dir.mkdir(exist_ok=True)
    ensure_write_permissions(temp_dir)
    shutil.copy(file_path, temp_dir / "main.c")

    makefile_content = f"""
include $(N64_INST)/include/n64.mk

src = main.c

all: {base_name}.z64

{base_name}.z64: N64_ROM_TITLE = "{base_name}"

clean:
    rm -f *.v64 *.z64 *.elf *.dfs $(builddir)/*.o

.PHONY: all clean
"""
    (temp_dir / "Makefile").write_text(makefile_content)

    os.chdir(temp_dir)
    try:
        subprocess.check_call(["make"], env=os.environ.copy())
        output_z64 = temp_dir / f"{base_name}.z64"
        final_path = Path.cwd() / output_z64.name
        shutil.move(output_z64, final_path)
        print(f"\nCompilation successful! Output: {final_path}")
    except subprocess.CalledProcessError as e:
        print("N64 Compilation failed.")
        print(e.output)

def main():
    while True:
        display_menu()
        choice = input("Select (1-3): ").strip()
        if choice == '1':
            file_path = get_file_path()
            format_choice = choose_format()
            if format_choice == '5':
                compile_n64(file_path)
            else:
                compile_host(file_path, format_choice)
        elif choice == '2':
            setup_n64_toolchain()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 3.")

if __name__ == "__main__":
    main()