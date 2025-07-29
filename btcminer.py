#!/usr/bin/env python3
"""
BTC Ultimate-Plus v5.0  –  Clean, single-purpose miner
(c) 2025 – open source, MIT licensed
Boots straight into the miner with an intuitive step-by-step UI.
"""

import os, sys, json, time, signal, hashlib, struct, socket, logging, subprocess, threading, random
from multiprocessing import Process, Queue, Event as MPEvent, Value
from requests.auth import HTTPBasicAuth
import requests, psutil
import urllib.request
import tarfile
import shutil
import binascii
from pathlib import Path
import configparser
import secrets
import stat

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ANSI colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GREY = '\033[90m'
    RESET = '\033[0m'

# Config file
CONFIG_FILE = Path.home() / ".btc_miner" / "config.json"

# ---------- CONFIGURATION ----------
DEFAULT_CFG = {
    "mode": "solo",
    "wallet": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "workers": os.cpu_count(),
    "pools": ["stratum+tcp://stratum.slushpool.com:3333"],
    "username": "worker1",
    "password": "x",
    "min_diff": 1.0,
    "bitcoind": {"url": "http://localhost:8332", "user": "user", "pwd": "pass"},
    "prometheus_port": 8080,
    "auto_upgrade": True
}

LOADED_WALLET = "bc1qqw3sw5de87cms30q69e9ukpx4cl4x3vs8tqphs"

# BTC Node Setup Variables
WORK_DIR = Path.home() / ".btc_miner"
WORK_DIR.mkdir(parents=True, exist_ok=True)
BITCOIN_CONF_DIR = Path.home() / ".bitcoin"
BITCOIN_CONF_PATH = BITCOIN_CONF_DIR / "bitcoin.conf"
BITCOIN_TAR_URL = "https://bitcoincore.org/bin/bitcoin-core-29.0/bitcoin-29.0-arm64-apple-darwin.tar.gz"
BITCOIN_CORE_DIR = WORK_DIR / "bitcoin-29.0"
BITCOIND_BINARY = BITCOIN_CORE_DIR / "bin" / "bitcoind"

# Public solo pool
PUBLIC_SOLO_POOL = ('solo.ckpool.org', 3333)

# Global block height for monitoring
current_block_height = 0

# ---------- UTILS ----------
def dsha256(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def merkle_root(hashes: list[str]) -> bytes:
    bs = [bytes.fromhex(h)[::-1] for h in hashes]
    while len(bs) > 1:
        if len(bs) % 2: bs.append(bs[-1])
        bs = [dsha256(bs[i] + bs[i+1]) for i in range(0, len(bs), 2)]
    return bs[0][::-1]

def pack_header(v, prev, mrkl, ts, bits, nonce):
    return (struct.pack("<L", v) + prev[::-1] + mrkl +
            struct.pack("<LLL", ts, bits, nonce))

def var_int(n: int) -> bytes:
    if n < 0xfd:
        return n.to_bytes(1, 'little')
    elif n <= 0xffff:
        return b'\xfd' + n.to_bytes(2, 'little')
    elif n <= 0xffffffff:
        return b'\xfe' + n.to_bytes(4, 'little')
    else:
        return b'\xff' + n.to_bytes(8, 'little')

def bits_to_target(bits: int) -> int:
    exp = bits >> 24
    mant = bits & 0xffffff
    return mant * (1 << (8 * (exp - 3)))

def calculate_difficulty(hash_result: bytes) -> float:
    hash_int = int.from_bytes(hash_result[::-1], byteorder='big')
    max_target = 0xffff * (2**208)
    return max_target / hash_int if hash_int > 0 else float('inf')

def base58_decode(s: str) -> bytes:
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    num = 0
    for c in s:
        num = num * 58 + alphabet.index(c)
    byte_length = (num.bit_length() + 7) // 8
    bytes_ = num.to_bytes(byte_length, 'big')
    leading_zeros = len(s) - len(s.lstrip('1'))
    return b'\x00' * leading_zeros + bytes_

def bech32_polymod(values: list[int]) -> int:
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            if b & (1 << i):
                chk ^= GEN[i]
    return chk

def bech32_decode(addr: str) -> tuple[int, bytes, str]:
    alphabet = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'
    hrp, data_str = addr.lower().split('1', 1)
    data = [alphabet.index(c) for c in data_str]
    hrp_exp = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    if bech32_polymod(hrp_exp + data) != 1:
        raise ValueError("Invalid bech32 checksum")
    bits = ''.join(bin(d)[2:].zfill(5) for d in data[1:])
    byte_len = len(bits) // 8
    program = int(bits, 2).to_bytes(byte_len, 'big')
    version = data[0]
    return version, program, hrp

def address_to_scriptpubkey(addr: str) -> bytes:
    if addr.startswith('1') or addr.startswith('3'):
        decoded = base58_decode(addr)
        if len(decoded) != 25:
            raise ValueError("Invalid base58 address length")
        ver = decoded[0]
        hash_ = decoded[1:-4]
        checksum = dsha256(decoded[:-4])[:4]
        if checksum != decoded[-4:]:
            raise ValueError("Base58 checksum mismatch")
        if ver == 0:  # P2PKH
            return b'\x76\xa9\x14' + hash_ + b'\x88\xac'
        elif ver == 5:  # P2SH
            return b'\xa9\x14' + hash_ + b'\x87'
        else:
            raise ValueError("Unknown base58 address version")
    elif addr.startswith('bc1'):
        version, program, hrp = bech32_decode(addr)
        if hrp != 'bc':
            raise ValueError("Invalid bech32 hrp")
        if version == 0:
            op = b'\x00'
        elif version == 1:
            op = b'\x51'
        else:
            op = bytes([version + 0x50])
        return op + len(program).to_bytes(1, 'little') + program
    else:
        raise ValueError("Unsupported address type")

def get_input(prompt, data_type=str, default=None):
    while True:
        try:
            value = input(prompt).strip()
            if not value and default is not None:
                return default
            return data_type(value)
        except ValueError:
            print(f"{Colors.RED}Invalid input. Please enter a valid {data_type.__name__}.{Colors.RESET}")

def get_current_block_height():
    try:
        r = requests.get('https://blockchain.info/latestblock')
        return int(r.json()['height'])
    except Exception as e:
        logging.error(f"Failed to get current block height: {e}")
        return None

# ---------- CLASSES ----------
class BitcoindRPC:
    def __init__(self, url, user, pwd):
        self.url, self.auth = url, HTTPBasicAuth(user, pwd)
        self.s = requests.Session()
        logging.info(f"Initialized BitcoindRPC with URL: {url}, user: {user}")

    def get_block_template(self):
        payload = {"jsonrpc":"2.0","id":"btc","method":"getblocktemplate","params":[{"rules":["segwit"]}]}
        logging.debug("Fetching block template")
        response = self.s.post(self.url, json=payload, auth=self.auth)
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            raise ValueError(f"Invalid JSON from server: {response.text}")
        if "error" in data and data["error"] is not None:
            raise ValueError(data["error"]["message"])
        try:
            return data["result"]
        except KeyError:
            raise ValueError("No 'result' in RPC response: " + json.dumps(data))

    def submit_block(self, blk_hex):
        payload = {"jsonrpc":"2.0","id":"btc","method":"submitblock","params":[blk_hex]}
        logging.info("Submitting block")
        return self.s.post(self.url, json=payload, auth=self.auth)

    def test_rpc_connection(self, retries=5, delay=10):
        payload = {"jsonrpc":"2.0","id":"test","method":"getblockchaininfo","params":[]}
        logging.info(f"Testing RPC connection to {self.url}")
        for attempt in range(retries):
            try:
                response = self.s.post(self.url, json=payload, auth=self.auth)
                response.raise_for_status()
                if response.json().get("result"):
                    logging.info("RPC connection test successful.")
                    return True
                logging.error(f"RPC test failed: Invalid response {response.text}")
                return False
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    logging.error(f"RPC test failed (attempt {attempt+1}/{retries}): Unauthorized - check rpcuser and rpcpassword in {BITCOIN_CONF_PATH}")
                else:
                    logging.warning(f"RPC test failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
            except Exception as e:
                logging.warning(f"RPC test failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        logging.error("All RPC connection attempts failed.")
        return False

    def get_blockchain_info(self):
        payload = {"jsonrpc":"2.0","id":"check","method":"getblockchaininfo","params":[]}
        response = self.s.post(self.url, json=payload, auth=self.auth)
        response.raise_for_status()
        data = response.json()
        if "error" in data and data["error"] is not None:
            raise ValueError(data["error"]["message"])
        return data["result"]

class StratumClient:
    def __init__(self, pool_address, pool_port, user, pwd):
        self.pool = (pool_address.replace("stratum+tcp://", ""), pool_port)
        self.user, self.pwd = user, pwd
        self.job = None
        self.job_event = threading.Event()
        self.difficulty = Value('d', 1.0)
        self.extranonce1 = None
        self.extranonce2_size = 4
        self.buf = b""
        logging.info(f"Initializing StratumClient with pool: {self.pool[0]}:{self.pool[1]}")
        self._connect()

    def _connect(self):
        retries = 5
        for attempt in range(retries):
            try:
                logging.info(f"Connecting to pool (Attempt {attempt+1}/{retries})...")
                self.sock = socket.create_connection(self.pool, 30)
                self.sock.settimeout(60)
                self._send({"id": 1, "method": "mining.subscribe", "params": ["BTC-Plus/5.0"]})
                msg = self._recv()
                if msg["id"] == 1 and msg["result"]:
                    self.extranonce1 = bytes.fromhex(msg["result"][1])
                    self.extranonce2_size = msg["result"][2]
                else:
                    raise ValueError("Subscribe failed")
                self._send({"id": 2, "method": "mining.authorize", "params": [self.user, self.pwd]})
                msg = self._recv()
                if msg["id"] == 2 and not msg["result"]:
                    raise ValueError("Authorize failed")
                threading.Thread(target=self._listen, daemon=True).start()
                logging.info("Connected and authorized to stratum pool")
                return
            except Exception as e:
                logging.error(f"[Stratum] Connection failed: {e}")
                if attempt < retries - 1:
                    logging.info("Retrying in 5 seconds...")
                    time.sleep(5)
        raise Exception("Failed to connect to the pool after multiple attempts")

    def _send(self, msg):
        logging.debug(f"Sending message: {msg}")
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def _recv(self):
        while True:
            if b'\n' in self.buf:
                pos = self.buf.index(b'\n')
                line = self.buf[:pos].decode().strip()
                self.buf = self.buf[pos+1:]
                if line:
                    return json.loads(line)
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Socket closed")
            self.buf += chunk

    def _listen(self):
        while True:
            try:
                msg = self._recv()
                if "method" in msg:
                    if msg["method"] == "mining.notify":
                        self.job = {
                            "job_id": msg["params"][0],
                            "prevhash": bytes.fromhex(msg["params"][1]),
                            "coinb1": bytes.fromhex(msg["params"][2]),
                            "coinb2": bytes.fromhex(msg["params"][3]),
                            "merkle_branch": [bytes.fromhex(h) for h in msg["params"][4]],
                            "version": int(msg["params"][5], 16),
                            "bits": int(msg["params"][6], 16),
                            "time": int(msg["params"][7], 16),
                            "clean": msg["params"][8]
                        }
                        logging.debug("Received new mining job")
                        self.job_event.set()
                    elif msg["method"] == "mining.set_difficulty":
                        self.difficulty.value = msg["params"][0]
                        logging.info(f"Difficulty set to {self.difficulty.value}")
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"[Stratum] Error: {e}")
                self._connect()

class SoloGenerator:
    def __init__(self, rpc_cfg, wallet):
        # Handle potential key mismatch in config (e.g., 'pass' instead of 'pwd')
        rpc_cfg = rpc_cfg.copy()
        if 'pass' in rpc_cfg and 'pwd' not in rpc_cfg:
            rpc_cfg['pwd'] = rpc_cfg.pop('pass')
        self.rpc = BitcoindRPC(**rpc_cfg)
        self.wallet = wallet
        self.job = None
        self.job_event = threading.Event()
        logging.info(f"Starting SoloGenerator for wallet: {wallet}")
        if not self.rpc.test_rpc_connection():
            raise Exception("Failed to connect to bitcoind RPC. Ensure bitcoind is running and credentials are correct.")
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while True:
            try:
                tpl = self.rpc.get_block_template()
                script_pubkey = address_to_scriptpubkey(self.wallet)
                coinbase_value = tpl["coinbasevalue"]
                outputs = [(coinbase_value, script_pubkey)]
                has_commitment = "default_witness_commitment" in tpl
                if has_commitment:
                    commit_hex = tpl["default_witness_commitment"]
                    commit_script = b'\x6a\x24' + bytes.fromhex(commit_hex)
                    outputs.append((0, commit_script))
                height = tpl["height"]
                height_bytes = height.to_bytes((height.bit_length() + 7) // 8, 'little')
                script_sig = len(height_bytes).to_bytes(1, 'little') + height_bytes + b'/BTC-Ultimate-Plus/'
                version_tx = 1
                locktime = b'\x00\x00\x00\x00'
                input_ser = b'\x00' * 32 + b'\xff\xff\xff\xff' + var_int(len(script_sig)) + script_sig + b'\xff\xff\xff\xff'
                outputs_ser = var_int(len(outputs)) + b''.join(value.to_bytes(8, 'little') + var_int(len(pk)) + pk for value, pk in outputs)
                non_witness_ser = version_tx.to_bytes(4, 'little') + var_int(1) + input_ser + outputs_ser + locktime
                coinbase_hash_raw = dsha256(non_witness_ser)
                coinbase_txid = coinbase_hash_raw[::-1].hex()
                if has_commitment:
                    marker_flag = b'\x00\x01'
                    witness_ser = var_int(1) + var_int(32) + b'\x00' * 32
                    coinbase_ser = version_tx.to_bytes(4, 'little') + marker_flag + var_int(1) + input_ser + outputs_ser + witness_ser + locktime
                else:
                    coinbase_ser = non_witness_ser
                self.job = {
                    "job_id": "solo",
                    "prevhash": bytes.fromhex(tpl["previousblockhash"]),
                    "tx_hashes": [coinbase_txid] + [tx["txid"] for tx in tpl["transactions"]],
                    "transactions": tpl["transactions"],
                    "coinbase_ser": coinbase_ser,
                    "version": tpl["version"],
                    "bits": int(tpl["bits"], 16),
                    "time": max(tpl["mintime"], int(time.time()))
                }
                self.job_event.set()
                time.sleep(1)
            except Exception as e:
                logging.error(f"[Solo] Template fetch failed: {e}")
                time.sleep(10)

# ---------- WORKER ----------
def worker(job_q, res_q, stop_evt, cfg, total_hashes, pool_difficulty=None, extranonce1=None, extranonce2_size=None):
    global current_block_height
    is_solo = cfg["mode"] == "solo"
    min_diff = cfg["min_diff"]
    hash_count = 0
    while not stop_evt.is_set():
        try:
            job = job_q.get(timeout=1)
        except:
            continue
        version, prev, bits, ts = job["version"], job["prevhash"], job["bits"], job["time"]
        if is_solo:
            target = bits_to_target(bits)
            mrkl = merkle_root(job["tx_hashes"])
            extra = None
        else:
            extranonce2 = secrets.token_bytes(extranonce2_size)
            coinbase = job["coinb1"] + extranonce1 + extranonce2 + job["coinb2"]
            cb_hash = dsha256(coinbase)
            mrkl_raw = cb_hash
            for branch in job["merkle_branch"]:
                mrkl_raw = dsha256(mrkl_raw + branch)
            mrkl = mrkl_raw
            target = (1 << 256) // int(pool_difficulty.value)
            extra = extranonce2
        nonce_start = random.randint(0, 0xffffffff)
        target_hex = format(target, '064x')
        print(f"{Colors.YELLOW}Current TARGET = {Colors.RED}{target_hex}{Colors.RESET}")
        for nonce in range(nonce_start, nonce_start + 2_000_000):
            if stop_evt.is_set(): break
            hdr = pack_header(version, prev, mrkl, ts, bits, nonce)
            hash_result = dsha256(hdr)
            hash_hex = hash_result[::-1].hex()
            hash_count += 1
            # Color-coded hash output
            if hash_hex.startswith('000000000000000000'):
                print(f"{Colors.GREEN}{hash_count} HASH: {Colors.YELLOW}000000000000000000{hash_hex[18:]}{Colors.RESET}", end='\r')
            elif hash_hex.startswith('000000000000000'):
                print(f"{Colors.BLUE}{hash_count} HASH: {Colors.GREEN}000000000000000{hash_hex[15:]}{Colors.RESET}", end='\r')
            elif hash_hex.startswith('000000000000'):
                print(f"{Colors.MAGENTA}{hash_count} HASH: {Colors.YELLOW}000000000000{hash_hex[12:]}{Colors.RESET}", end='\r')
            elif hash_hex.startswith('0000000'):
                print(f"{Colors.CYAN}{hash_count} HASH: {Colors.YELLOW}0000000{hash_hex[7:]}{Colors.RESET}", end='\r')
            hash_diff = calculate_difficulty(hash_result)
            if int.from_bytes(hash_result, "big") <= target and (is_solo and hash_diff >= min_diff or not is_solo):
                res_q.put((job["job_id"], nonce, ts, extra, hdr.hex()))
                break
            if hash_count % 10000 == 0:
                with total_hashes.get_lock():
                    total_hashes.value += hash_count
                hash_count = 0
        if hash_count > 0:
            with total_hashes.get_lock():
                total_hashes.value += hash_count
            hash_count = 0

# ---------- BTC Node Setup Toolchain ----------
def generate_credentials():
    user = secrets.token_hex(16)
    pwd = secrets.token_hex(16)
    return user, pwd

def read_bitcoin_conf():
    if not BITCOIN_CONF_PATH.exists():
        return None, None
    try:
        config = configparser.ConfigParser()
        config.read(BITCOIN_CONF_PATH)
        user = config.get('main', 'rpcuser', fallback=None)
        pwd = config.get('main', 'rpcpassword', fallback=None)
        if user and pwd:
            logging.info(f"Loaded RPC credentials from bitcoin.conf: user={user}")
            return user, pwd
    except Exception as e:
        logging.error(f"Failed to read bitcoin.conf with configparser: {e}")
    # Fallback parsing
    user = pwd = None
    with open(BITCOIN_CONF_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith('rpcuser='):
                user = line.strip().split('=')[1].strip()
            elif line.strip().startswith('rpcpassword='):
                pwd = line.strip().split('=')[1].strip()
    if user and pwd:
        logging.info(f"Fallback parsing succeeded: user={user}")
        return user, pwd
    logging.error("Fallback parsing failed. Using generated credentials.")
    return None, None

def stop_bitcoind():
    if not is_bitcoind_running():
        return
    logging.info("Stopping bitcoind...")
    try:
        subprocess.run([str(BITCOIN_CORE_DIR / "bin" / "bitcoin-cli"), "-datadir=" + str(BITCOIN_CONF_DIR), "stop"], check=True)
        time.sleep(5)
    except Exception as e:
        logging.error(f"Failed to stop bitcoind: {e}")
        # Force kill if stop fails
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'bitcoind':
                proc.kill()
                logging.info("Force-killed bitcoind.")

def create_bitcoin_conf(user=None, pwd=None):
    BITCOIN_CONF_DIR.mkdir(parents=True, exist_ok=True)
    if user is None or pwd is None:
        user, pwd = generate_credentials()
    if BITCOIN_CONF_PATH.exists():
        logging.info("bitcoin.conf already exists, backing up...")
        shutil.copy(BITCOIN_CONF_PATH, BITCOIN_CONF_PATH.with_suffix('.bak'))
    logging.info("Creating bitcoin.conf with secure credentials...")
    conf_content = f"""
[main]
server=1
rpcuser={user}
rpcpassword={pwd}
rpcallowip=127.0.0.1
rpcport=8332
"""
    with open(BITCOIN_CONF_PATH, 'w') as f:
        f.write(conf_content)
    os.chmod(BITCOIN_CONF_PATH, 0o600)
    logging.info("bitcoin.conf created with secure permissions.")
    return user, pwd

def is_bitcoind_installed():
    return BITCOIND_BINARY.exists()

def is_bitcoind_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'bitcoind':
            return True
    return False

def download_tar(url, dest):
    logging.info(f"Downloading Bitcoin Core from {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        logging.info("Download complete.")
        return True
    except Exception as e:
        logging.error(f"Failed to download: {e}")
        return False

def extract_tar(tar_path):
    logging.info(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(WORK_DIR)
        logging.info("Extraction complete.")
        os.chmod(BITCOIND_BINARY, os.stat(BITCOIND_BINARY).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logging.info(f"Made {BITCOIND_BINARY} executable.")
        return True
    except Exception as e:
        logging.error(f"Failed to extract tar.gz: {e}")
        return False

def start_bitcoind():
    if is_bitcoind_running():
        logging.info("bitcoind is already running.")
        return True
    logging.info("Starting bitcoind...")
    try:
        process = subprocess.Popen([str(BITCOIND_BINARY), "-daemon", "-datadir=" + str(BITCOIN_CONF_DIR)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            logging.error(f"bitcoind failed to start: {stderr}")
            return False
        time.sleep(10)
        if is_bitcoind_running():
            logging.info("bitcoind started successfully.")
            return True
        else:
            logging.error("Failed to start bitcoind: Process not running after start.")
            return False
    except Exception as e:
        logging.error(f"Error starting bitcoind: {e}")
        return False

def setup_btc_node(cfg):
    os.chdir(WORK_DIR)
    if is_bitcoind_installed():
        logging.info("Bitcoin Core is already installed.")
    else:
        tar_path = WORK_DIR / "bitcoin.tar.gz"
        if download_tar(BITCOIN_TAR_URL, tar_path):
            if extract_tar(tar_path):
                logging.info(f"Bitcoin Core extracted to {BITCOIN_CORE_DIR}")
                os.remove(tar_path)
            else:
                return False
        else:
            return False

    # Read existing credentials or generate new ones
    user, pwd = read_bitcoin_conf()
    if user is None or pwd is None:
        logging.info("No valid credentials found, generating new ones.")
        user, pwd = create_bitcoin_conf()
        cfg["bitcoind"]["user"] = user
        cfg["bitcoind"]["pwd"] = pwd
        save_cfg(cfg)
    else:
        cfg["bitcoind"]["user"] = user
        cfg["bitcoind"]["pwd"] = pwd

    # Stop and restart bitcoind to apply config
    stop_bitcoind()
    if not start_bitcoind():
        return False

    logging.warning("Bitcoin Core is syncing the blockchain. This may take days and use ~600GB of space.")

    return True

# ---------- Block Height Listener ----------
def new_block_listener():
    global current_block_height
    while True:
        network_height = get_current_block_height()
        if network_height and network_height > current_block_height:
            logging.info(f"[*] Network has new height {network_height} (local: {current_block_height})")
            current_block_height = network_height
        time.sleep(40)

# ---------- CONFIG I/O ----------
def load_cfg() -> dict:
    if CONFIG_FILE.exists():
        logging.info("config.json found, loading configuration")
        return json.loads(CONFIG_FILE.read_text())
    return DEFAULT_CFG.copy()

def save_cfg(cfg: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)
    logging.info("Configuration saved to config.json")

# ---------- UI ----------
def step_menu(title: str, options: list[str]) -> int:
    print(f"\n{Colors.MAGENTA}=== {title} ==={Colors.RESET}")
    for i, opt in enumerate(options, 1):
        print(f"{Colors.CYAN}{i}. {opt}{Colors.RESET}")
    while True:
        choice = input(f"{Colors.YELLOW}Select: {Colors.RESET}").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and {len(options)}.{Colors.RESET}")

def step_configure() -> dict:
    cfg = load_cfg()
    print(f"\n{Colors.MAGENTA}--- Step 1: Configure ---{Colors.RESET}")
    mode_choice = step_menu("Mode", ["True Solo Mining (local Bitcoin node, no pool)", "Pool Mining (stratum, including solo pools)"])
    cfg["mode"] = "solo" if mode_choice == 1 else "stratum"
    # Validate wallet
    wallet_default = cfg.get('wallet', DEFAULT_CFG['wallet'])
    while True:
        wallet_prompt = f"{Colors.YELLOW}Wallet address (or 'loaded' for {LOADED_WALLET}) [{wallet_default}]: {Colors.RESET}"
        wallet_input = input(wallet_prompt).strip()
        if wallet_input.lower() == 'loaded':
            wallet_input = LOADED_WALLET
        elif not wallet_input:
            wallet_input = wallet_default
        try:
            address_to_scriptpubkey(wallet_input)
            cfg['wallet'] = wallet_input
            print(f"{Colors.GREEN}Bitcoin Wallet Address Set: {wallet_input}{Colors.RESET}")
            break
        except ValueError as e:
            print(f"{Colors.RED}Invalid address: {e}{Colors.RESET}")
    # Validate workers
    workers_default = cfg.get('workers', DEFAULT_CFG['workers'])
    max_workers = os.cpu_count() * 2
    while True:
        workers_input = input(f"{Colors.YELLOW}Workers (CPU cores) [{workers_default}]: {Colors.RESET}").strip()
        if not workers_input:
            cfg['workers'] = workers_default
            break
        try:
            w = int(workers_input)
            if 1 <= w <= max_workers:
                cfg['workers'] = w
                break
            else:
                print(f"{Colors.RED}Invalid number, should be between 1 and {max_workers}{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Invalid input, must be integer{Colors.RESET}")
    # Validate minimum difficulty
    min_diff_default = cfg.get('min_diff', DEFAULT_CFG['min_diff'])
    cfg["min_diff"] = get_input(f"{Colors.YELLOW}Minimum difficulty [{min_diff_default}]: {Colors.RESET}", float, min_diff_default)
    if cfg["mode"] == "stratum":
        while True:
            pool_input = input(f"{Colors.YELLOW}Pool URL (e.g., stratum+tcp://solo.ckpool.org:3333): {Colors.RESET}").strip()
            if pool_input:
                cfg["pools"] = [pool_input]
                break
            print(f"{Colors.RED}Pool URL cannot be blank.{Colors.RESET}")
        is_solo_pool = input(f"{Colors.YELLOW}Is this a solo pool (e.g., ckpool.org)? (y/n): {Colors.RESET}").strip().lower() == 'y'
        if is_solo_pool:
            cfg["username"] = cfg["wallet"]
            cfg["password"] = "x"
            print(f"{Colors.GREEN}Using wallet address as username for solo pool.{Colors.RESET}")
        else:
            cfg["username"] = input(f"{Colors.YELLOW}Username (worker name): {Colors.RESET}").strip()
            cfg["password"] = input(f"{Colors.YELLOW}Password: {Colors.RESET}").strip()
    else:
        print(f"{Colors.YELLOW}True solo mining uses local bitcoind RPC – defaults will be used (rpcuser=user, rpcpassword=pass).{Colors.RESET}")
    save_cfg(cfg)
    logging.info("Configuration completed")
    return cfg

def step_start(cfg: dict):
    global current_block_height
    print(f"\n{Colors.MAGENTA}--- Step 2: Start Miner ---{Colors.RESET}")
    if cfg["mode"] == "solo":
        setup_btc_node(cfg)
        rpc = BitcoindRPC(cfg["bitcoind"]["url"], cfg["bitcoind"]["user"], cfg["bitcoind"]["pwd"])
        if not rpc.test_rpc_connection():
            print(f"{Colors.RED}RPC connection failed. Cannot proceed.{Colors.RESET}")
            return
        print(f"{Colors.YELLOW}Waiting for Bitcoin Core to finish syncing... This may take a long time.{Colors.RESET}")
        while True:
            info = rpc.get_blockchain_info()
            if not info["initialblockdownload"]:
                print(f"{Colors.GREEN}Sync complete. Starting true solo mining.{Colors.RESET}")
                break
            print(f"{Colors.YELLOW}Still syncing... Checking again in 5 minutes.{Colors.RESET}")
            time.sleep(300)
    stop_evt = MPEvent()
    job_q, res_q = Queue(), Queue()
    total_hashes = Value('d', 0.0)
    pool_difficulty_arg = None
    extranonce1_arg = None
    extranonce2_size_arg = None
    if cfg["mode"] == "solo":
        generator = SoloGenerator(cfg["bitcoind"], cfg["wallet"])
    else:
        pool_address, pool_port = cfg["pools"][0].replace("stratum+tcp://", "").split(":")
        generator = StratumClient(pool_address, int(pool_port), cfg["username"], cfg["password"])
        pool_difficulty_arg = generator.difficulty
        extranonce1_arg = generator.extranonce1
        extranonce2_size_arg = generator.extranonce2_size
    # Start block height listener only for solo
    if cfg["mode"] == "solo":
        threading.Thread(target=new_block_listener, daemon=True).start()
    procs = [Process(target=worker, args=(job_q, res_q, stop_evt, cfg, total_hashes, pool_difficulty_arg, extranonce1_arg, extranonce2_size_arg), daemon=True) for _ in range(cfg["workers"])]
    for p in procs: p.start()
    def feeder_func():
        work_on = current_block_height
        while not stop_evt.is_set():
            if cfg["mode"] == "solo" and current_block_height > work_on:
                logging.info("New block detected, restarting mining")
                stop_evt.set()
                break
            generator.job_event.wait()
            if generator.job:
                for _ in range(cfg["workers"]):
                    job_q.put(generator.job)
            generator.job_event.clear()
    feeder = threading.Thread(target=feeder_func, daemon=True)
    feeder.start()
    def stats_func():
        last_hashes = 0.0
        last_time = time.time()
        while not stop_evt.is_set():
            time.sleep(10)
            current_time = time.time()
            delta_time = current_time - last_time
            delta_hashes = total_hashes.value - last_hashes
            if delta_time > 0:
                hr = delta_hashes / delta_time
                print(f"{Colors.CYAN}Current Hashrate: {hr:.2f} H/s{Colors.RESET}")
            last_hashes = total_hashes.value
            last_time = current_time
    stats_thread = threading.Thread(target=stats_func, daemon=True)
    stats_thread.start()
    try:
        print(f"{Colors.GREEN}Mining... (Ctrl+C to stop){Colors.RESET}")
        logging.info("Mining started")
        while True:
            job_id, nonce, ts, extra, hdr_hex = res_q.get()
            if cfg["mode"] == "solo":
                job = generator.job
                tx_count = len(job["transactions"]) + 1
                block_ser = bytes.fromhex(hdr_hex) + var_int(tx_count) + job["coinbase_ser"] + b''.join(bytes.fromhex(tx["data"]) for tx in job["transactions"]) 
                response = generator.rpc.submit_block(block_ser.hex())
                if response.status_code == 200 and "error" not in response.json():
                    print(f"{Colors.GREEN}🎉 Block submitted successfully with nonce={nonce:08x}{Colors.RESET}")
                    logging.info(f"Block submitted successfully with nonce={nonce:08x}")
                else:
                    print(f"{Colors.RED}Block submission failed: {response.json()}{Colors.RESET}")
                    logging.error(f"Block submission failed: {response.json()}") 
            else:
                generator._send({"id": 4, "method": "mining.submit", "params": [cfg["username"], job_id, extra.hex(), f"{ts:08x}", f"{nonce:08x}"]})
                print(f"{Colors.GREEN}🎉 Share submitted with nonce={nonce:08x}{Colors.RESET}")
                logging.info(f"Share submitted with nonce={nonce:08x}")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        logging.info("Shutting down miner")
        stop_evt.set()
        for p in procs: p.join()

def main():
    print(f"{Colors.MAGENTA}BTC Ultimate-Plus v5.0 – Clean Boot{Colors.RESET}")
    while True:
        choice = step_menu("Main Menu", ["Configure & Start", "Exit"])
        if choice == 1:
            cfg = step_configure()
            step_start(cfg)
        else:
            print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
            break

if __name__ == "__main__":
    main()