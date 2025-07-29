#!/usr/bin/env python3
"""
DOGE Ultimate-Plus v1.0  –  Clean, single-purpose miner
(c) 2025 – open source, MIT licensed
Boots straight into the miner with an intuitive step-by-step UI.
"""

import os, sys, json, time, signal, hashlib, struct, socket, logging, subprocess, threading, random
import hmac
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
CONFIG_FILE = Path.home() / ".doge_miner" / "config.json"

# ---------- CONFIGURATION ----------
DEFAULT_CFG = {
    "mode": "pool",
    "doge_wallet": "D6Zed9c6JfhhvKNF9g1VB6me6FukRm1bY3",
    "ltc_wallet": "LTC_ADDRESS_HERE",
    "workers": os.cpu_count(),
    "pools": ["stratum+tcp://stratum.aikapool.com:7915"],
    "username": "worker1",
    "password": "x",
    "min_diff": 1.0,
    "dogecoind": {"url": "http://localhost:22555", "user": "user", "pwd": "pass"},
    "prometheus_port": 8080,
    "auto_upgrade": True
}

LOADED_DOGE_WALLET = "DB9hq9pJitN3M44xPxjHKctuQZaimTuyed"
LOADED_LTC_WALLET = "ltc1qgkls5dj0r267z2f8jscs65lrpk44ly4qtzfltd"

# Default merged pool (ViaBTC for LTC+DOGE merged)
DEFAULT_MERGED_POOL = "stratum+tcp://ltc.viabtc.com:3333"

# DOGE Node Setup Variables
WORK_DIR = Path.home() / ".doge_miner"
WORK_DIR.mkdir(parents=True, exist_ok=True)
DOGECOIN_CONF_DIR = Path.home() / ".dogecoin"
DOGECOIN_CONF_PATH = DOGECOIN_CONF_DIR / "dogecoin.conf"
DOGECOIN_TAR_URL = "https://github.com/dogecoin/dogecoin/releases/download/v1.14.9/dogecoin-1.14.9.tar.gz"
DOGECOIN_CORE_DIR = WORK_DIR / "dogecoin-1.14.9"
DOGECOIND_BINARY = DOGECOIN_CORE_DIR / "src" / "dogecoind"

# Global block height for monitoring
current_block_height = 0

# ---------- Scrypt Implementation ----------
def scrypt(password, salt, N, r, p, dkLen):
    def array_overwrite(source, source_start, dest, dest_start, length):
        for i in range(length):
            dest[dest_start + i] = source[source_start + i]

    def blockxor(source, source_start, dest, dest_start, length):
        for i in range(length):
            dest[dest_start + i] ^= source[source_start + i]

    def pbkdf2(passphrase, salt, count, dkLen, prf):
        size = 0
        blocks = []
        block_number = 0
        while size < dkLen:
            block_number += 1
            block = prf(passphrase, salt + struct.pack('>I', block_number))
            U = block
            for _ in range(2, 1 + count):
                U = prf(passphrase, U)
                block = bytes(b1 ^ b2 for b1, b2 in zip(block, U))
            blocks.append(block)
            size += len(block)
        return b''.join(blocks)[:dkLen]

    def integerify(B, Bi, r):
        Bi += (2 * r - 1) * 64
        return int.from_bytes(bytes(B[Bi:Bi+8]), 'little')

    def R(X, destination, a1, a2, b):
        a = (X[a1] + X[a2]) & 0xffffffff
        X[destination] ^= ((a << b) | (a >> (32 - b))) & 0xffffffff

    def salsa20_8(B):
        B32 = [int.from_bytes(B[i*4:(i+1)*4], 'little') for i in range(16)]
        x = list(B32)
        for _ in range(8, 0, -2):
            R(x, 4, 0, 12, 7); R(x, 8, 4, 0, 9); R(x, 12, 8, 4, 13); R(x, 0, 12, 8, 18)
            R(x, 9, 5, 1, 7); R(x, 13, 9, 5, 9); R(x, 1, 13, 9, 13); R(x, 5, 1, 13, 18)
            R(x, 14, 10, 6, 7); R(x, 2, 14, 10, 9); R(x, 6, 2, 14, 13); R(x, 10, 6, 2, 18)
            R(x, 3, 15, 11, 7); R(x, 7, 3, 15, 9); R(x, 11, 7, 3, 13); R(x, 15, 11, 7, 18)
            R(x, 1, 0, 3, 7); R(x, 2, 1, 0, 9); R(x, 3, 2, 1, 13); R(x, 0, 3, 2, 18)
            R(x, 6, 5, 4, 7); R(x, 7, 6, 5, 9); R(x, 4, 7, 6, 13); R(x, 5, 4, 7, 18)
            R(x, 11, 10, 9, 7); R(x, 8, 11, 10, 9); R(x, 9, 8, 11, 13); R(x, 10, 9, 8, 18)
            R(x, 12, 15, 14, 7); R(x, 13, 12, 15, 9); R(x, 14, 13, 12, 13); R(x, 15, 14, 13, 18)
        B32 = [(x[i] + B32[i]) & 0xffffffff for i in range(16)]
        return b''.join(struct.pack('<I', b) for b in B32)

    def blockmix_salsa8(BY, Bi, Yi, r):
        X = BY[Bi + (2 * r - 1) * 64 : Bi + (2 * r) * 64]
        for i in range(2 * r):
            blockxor(BY, Bi + i * 64, X, 0, 64)
            X = salsa20_8(bytes(X))
            X = list(X)
            array_overwrite(X, 0, BY, Yi + i * 64, 64)
        for i in range(r):
            array_overwrite(BY, Yi + i * 128, BY, Bi + i * 64, 64)
        for i in range(r):
            array_overwrite(BY, Yi + (i * 128 + 64), BY, Bi + (i + r) * 64, 64)

    def smix(B, Bi, r, N, V, XY):
        array_overwrite(B, Bi, XY, 0, 128 * r)
        for i in range(N):
            array_overwrite(XY, 0, V, i * (128 * r), 128 * r)
            blockmix_salsa8(XY, 0, 128 * r, r)
        for i in range(N):
            j = integerify(XY, 0, r) & (N - 1)
            blockxor(V, j * (128 * r), XY, 0, 128 * r)
            blockmix_salsa8(XY, 0, 128 * r, r)
        array_overwrite(XY, 0, B, Bi, 128 * r)

    if N < 2 or (N & (N - 1)): raise ValueError('Scrypt N must be a power of 2 greater than 1')

    prf = lambda k, m: hmac.new(key=k, msg=m, digestmod=hashlib.sha256).digest()

    PB = pbkdf2(password, salt, 1, p * 128 * r, prf)
    B = list(PB)

    XY = [0] * (256 * r)
    V = [0] * (128 * r * N)

    for i in range(p):
        smix(B, i * 128 * r, r, N, V, XY)

    return pbkdf2(password, bytes(B), 1, dkLen, prf)

def scrypt_hash(header):
    return scrypt(header, header, N=1024, r=1, p=1, dkLen=32)

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

def address_to_scriptpubkey(addr: str) -> bytes:
    decoded = base58_decode(addr)
    if len(decoded) != 25:
        raise ValueError("Invalid base58 address length")
    ver = decoded[0]
    hash_ = decoded[1:-4]
    checksum = dsha256(decoded[:-4])[:4]
    if checksum != decoded[-4:]:
        raise ValueError("Base58 checksum mismatch")
    if ver == 30:  # P2PKH for Dogecoin
        return b'\x76\xa9\x14' + hash_ + b'\x88\xac'
    elif ver == 22:  # P2SH for Dogecoin
        return b'\xa9\x14' + hash_ + b'\x87'
    else:
        raise ValueError("Unknown base58 address version for Dogecoin")

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
        r = requests.get('https://dogechain.info/api/v1/blockchain/height')
        return int(r.json()['height'])
    except Exception as e:
        logging.error(f"Failed to get current block height: {e}")
        return None

# ---------- CLASSES ----------
class DogecoindRPC:
    def __init__(self, url, user, pwd):
        self.url, self.auth = url, HTTPBasicAuth(user, pwd)
        self.s = requests.Session()
        logging.info(f"Initialized DogecoindRPC with URL: {url}, user: {user}")

    def get_block_template(self):
        payload = {"jsonrpc":"2.0","id":"doge","method":"getblocktemplate","params":[{"rules":["segwit"]}]}
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
        payload = {"jsonrpc":"2.0","id":"doge","method":"submitblock","params":[blk_hex]}
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
                    logging.error(f"RPC test failed (attempt {attempt+1}/{retries}): Unauthorized - check rpcuser and rpcpassword in {DOGECOIN_CONF_PATH}")
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
                resolved_ip = socket.gethostbyname(self.pool[0])
                logging.info(f"Resolved {self.pool[0]} to IP: {resolved_ip}")
                logging.info(f"Connecting to pool (Attempt {attempt+1}/{retries})...")
                self.sock = socket.create_connection(self.pool, 30)
                self.sock.settimeout(60)
                self._send({"id": 1, "method": "mining.subscribe", "params": ["DOGE-Plus/1.0"]})
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
        self.rpc = DogecoindRPC(**rpc_cfg)
        self.wallet = wallet
        self.job = None
        self.job_event = threading.Event()
        logging.info(f"Starting SoloGenerator for wallet: {wallet}")
        if not self.rpc.test_rpc_connection():
            raise Exception("Failed to connect to dogecoind RPC. Ensure dogecoind is running and credentials are correct.")
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
                script_sig = len(height_bytes).to_bytes(1, 'little') + height_bytes + b'/DOGE-Ultimate-Plus/'
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
            hash_result = scrypt_hash(hdr)
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

# ---------- DOGE Node Setup Toolchain ----------
def generate_credentials():
    user = secrets.token_hex(16)
    pwd = secrets.token_hex(16)
    return user, pwd

def read_dogecoin_conf():
    if not DOGECOIN_CONF_PATH.exists():
        return None, None
    try:
        config = configparser.ConfigParser()
        config.read(DOGECOIN_CONF_PATH)
        user = config.get('main', 'rpcuser', fallback=None)
        pwd = config.get('main', 'rpcpassword', fallback=None)
        if user and pwd:
            logging.info(f"Loaded RPC credentials from dogecoin.conf: user={user}")
            return user, pwd
    except Exception as e:
        logging.error(f"Failed to read dogecoin.conf with configparser: {e}")
    # Fallback parsing
    user = pwd = None
    with open(DOGECOIN_CONF_PATH, 'r') as f:
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

def stop_dogecoind():
    if not is_dogecoind_running():
        return
    logging.info("Stopping dogecoind...")
    try:
        subprocess.run([str(DOGECOIND_BINARY), "-datadir=" + str(DOGECOIN_CONF_DIR), "stop"], check=True)
        time.sleep(5)
    except Exception as e:
        logging.error(f"Failed to stop dogecoind: {e}")
        # Force kill if stop fails
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] == 'dogecoind':
                proc.kill()
                logging.info("Force-killed dogecoind.")

def create_dogecoin_conf(user=None, pwd=None):
    DOGECOIN_CONF_DIR.mkdir(parents=True, exist_ok=True)
    if user is None or pwd is None:
        user, pwd = generate_credentials()
    if DOGECOIN_CONF_PATH.exists():
        logging.info("dogecoin.conf already exists, backing up...")
        shutil.copy(DOGECOIN_CONF_PATH, DOGECOIN_CONF_PATH.with_suffix('.bak'))
    logging.info("Creating dogecoin.conf with secure credentials...")
    conf_content = f"""
[main]
server=1
rpcuser={user}
rpcpassword={pwd}
rpcallowip=127.0.0.1
rpcport=22555
"""
    with open(DOGECOIN_CONF_PATH, 'w') as f:
        f.write(conf_content)
    os.chmod(DOGECOIN_CONF_PATH, 0o600)
    logging.info("dogecoin.conf created with secure permissions.")
    return user, pwd

def is_dogecoind_installed():
    return DOGECOIND_BINARY.exists()

def is_dogecoind_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'dogecoind':
            return True
    return False

def download_tar(url, dest):
    logging.info(f"Downloading Dogecoin Core source from {url}...")
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
        return True
    except Exception as e:
        logging.error(f"Failed to extract tar.gz: {e}")
        return False

def start_dogecoind():
    if is_dogecoind_running():
        logging.info("dogecoind is already running.")
        return True
    logging.info("Starting dogecoind...")
    try:
        process = subprocess.Popen([str(DOGECOIND_BINARY), "-daemon", "-datadir=" + str(DOGECOIN_CONF_DIR)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            logging.error(f"dogecoind failed to start: {stderr}")
            return False
        time.sleep(10)
        if is_dogecoind_running():
            logging.info("dogecoind started successfully.")
            return True
        else:
            logging.error("Failed to start dogecoind: Process not running after start.")
            return False
    except Exception as e:
        logging.error(f"Error starting dogecoind: {e}")
        return False

def setup_doge_node(cfg):
    os.chdir(WORK_DIR)
    if is_dogecoind_installed():
        logging.info("Dogecoin Core is already installed.")
    else:
        tar_path = WORK_DIR / "dogecoin.tar.gz"
        if download_tar(DOGECOIN_TAR_URL, tar_path):
            if extract_tar(tar_path):
                logging.info(f"Dogecoin Core source extracted to {DOGECOIN_CORE_DIR}")
                os.remove(tar_path)
            else:
                return False
        else:
            return False

        # Install dependencies using Homebrew
        dependencies = ["autoconf", "automake", "libtool", "pkg-config", "boost", "libevent", "qt@5", "protobuf", "zeromq", "berkeley-db@4", "miniupnpc", "qrencode"]
        try:
            subprocess.check_call(["brew", "install"] + dependencies)
        except Exception as e:
            logging.error(f"Failed to install dependencies: {e}")
            return False

        # Compile Dogecoin Core
        os.chdir(DOGECOIN_CORE_DIR)
        try:
            subprocess.check_call(["./autogen.sh"])
            subprocess.check_call(["./configure", "--with-gui=qt5"])
            subprocess.check_call(["make"])
        except Exception as e:
            logging.error(f"Failed to compile Dogecoin Core: {e}")
            return False

        os.chmod(DOGECOIND_BINARY, os.stat(DOGECOIND_BINARY).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Read existing credentials or generate new ones
    user, pwd = read_dogecoin_conf()
    if user is None or pwd is None:
        logging.info("No valid credentials found, generating new ones.")
        user, pwd = create_dogecoin_conf()
        cfg["dogecoind"]["user"] = user
        cfg["dogecoind"]["pwd"] = pwd
        save_cfg(cfg)
    else:
        cfg["dogecoind"]["user"] = user
        cfg["dogecoind"]["pwd"] = pwd

    # Stop and restart dogecoind to apply config
    stop_dogecoind()
    if not start_dogecoind():
        return False

    logging.warning("Dogecoin Core is syncing the blockchain. This may take days and use ~50GB of space.")

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
    mode_choice = step_menu("Mode", ["True Solo Mining (local Dogecoin node, no pool)", "Pool Mining (stratum, including solo pools)", "Merged Mining (LTC pool with merged DOGE)"])
    if mode_choice == 1:
        cfg["mode"] = "solo"
    elif mode_choice == 2:
        cfg["mode"] = "pool"
    else:
        cfg["mode"] = "merged"
    # Validate doge wallet
    wallet_default = cfg.get('doge_wallet', DEFAULT_CFG['doge_wallet'])
    while True:
        wallet_prompt = f"{Colors.YELLOW}Dogecoin wallet address (or 'loaded' for {LOADED_DOGE_WALLET}) [{wallet_default}]: {Colors.RESET}"
        wallet_input = input(wallet_prompt).strip()
        if wallet_input.lower() == 'loaded':
            wallet_input = LOADED_DOGE_WALLET
        elif not wallet_input:
            wallet_input = wallet_default
        try:
            address_to_scriptpubkey(wallet_input)
            cfg['doge_wallet'] = wallet_input
            print(f"{Colors.GREEN}Dogecoin Wallet Address Set: {wallet_input}{Colors.RESET}")
            break
        except ValueError as e:
            print(f"{Colors.RED}Invalid address: {e}{Colors.RESET}")
    if cfg["mode"] == "merged":
        ltc_default = cfg.get('ltc_wallet', DEFAULT_CFG['ltc_wallet'])
        while True:
            ltc_prompt = f"{Colors.YELLOW}Litecoin wallet address (or 'loaded' for {LOADED_LTC_WALLET}) [{ltc_default}]: {Colors.RESET}"
            ltc_input = input(ltc_prompt).strip()
            if ltc_input.lower() == 'loaded':
                ltc_input = LOADED_LTC_WALLET
            elif not ltc_input:
                ltc_input = ltc_default
            # Basic validation for LTC address
            if ltc_input.startswith('L') or ltc_input.startswith('M') or ltc_input.startswith('ltc1'):
                cfg['ltc_wallet'] = ltc_input
                print(f"{Colors.GREEN}Litecoin Wallet Address Set: {ltc_input}{Colors.RESET}")
                break
            else:
                print(f"{Colors.RED}Invalid Litecoin address. Please enter a valid LTC address.{Colors.RESET}")
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
    if cfg["mode"] != "solo":
        if cfg["mode"] == "merged":
            while True:
                pool_prompt = f"{Colors.YELLOW}Merged Pool URL (or 'loaded' for {DEFAULT_MERGED_POOL}): {Colors.RESET}"
                pool_input = input(pool_prompt).strip()
                if pool_input.lower() == 'loaded':
                    pool_input = DEFAULT_MERGED_POOL
                if pool_input:
                    cfg["pools"] = [pool_input]
                    break
                print(f"{Colors.RED}Pool URL cannot be blank.{Colors.RESET}")
            cfg["username"] = cfg["ltc_wallet"]
            cfg["password"] = "d=" + cfg["doge_wallet"]
            print(f"{Colors.GREEN}Using merged mining pool with LTC as username and d=DOGE as password.{Colors.RESET}")
        else:
            while True:
                pool_prompt = f"{Colors.YELLOW}Pool URL (e.g., stratum+tcp://stratum.aikapool.com:7915 or 'loaded' for default): {Colors.RESET}"
                pool_input = input(pool_prompt).strip()
                if pool_input.lower() == 'loaded':
                    pool_input = DEFAULT_CFG["pools"][0]
                if pool_input:
                    cfg["pools"] = [pool_input]
                    break
                print(f"{Colors.RED}Pool URL cannot be blank.{Colors.RESET}")
            is_solo_pool = input(f"{Colors.YELLOW}Is this a solo pool? (y/n): {Colors.RESET}").strip().lower() == 'y'
            if is_solo_pool:
                cfg["username"] = cfg["doge_wallet"]
                cfg["password"] = "x"
                print(f"{Colors.GREEN}Using wallet address as username for solo pool.{Colors.RESET}")
            else:
                cfg["username"] = input(f"{Colors.YELLOW}Username (worker name): {Colors.RESET}").strip()
                cfg["password"] = input(f"{Colors.YELLOW}Password: {Colors.RESET}").strip()
    else:
        print(f"{Colors.YELLOW}True solo mining uses local dogecoind RPC – defaults will be used. Note: Solo mining DOGE without merged is not recommended.{Colors.RESET}")
    save_cfg(cfg)
    logging.info("Configuration completed")
    return cfg

def step_start(cfg: dict):
    global current_block_height
    print(f"\n{Colors.MAGENTA}--- Step 2: Start Miner ---{Colors.RESET}")
    if cfg["mode"] == "solo":
        if not setup_doge_node(cfg):
            print(f"{Colors.RED}Failed to set up Dogecoin node. Cannot proceed with solo mining.{Colors.RESET}")
            return
        rpc = DogecoindRPC(cfg["dogecoind"]["url"], cfg["dogecoind"]["user"], cfg["dogecoind"]["pwd"])
        if not rpc.test_rpc_connection():
            print(f"{Colors.RED}RPC connection failed. Cannot proceed.{Colors.RESET}")
            return
        print(f"{Colors.YELLOW}Waiting for Dogecoin Core to finish syncing... This may take a long time.{Colors.RESET}")
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
        generator = SoloGenerator(cfg["dogecoind"], cfg["doge_wallet"])
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
    print(f"{Colors.MAGENTA}DOGE Ultimate-Plus v1.0 – Clean Boot{Colors.RESET}")
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