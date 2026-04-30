"""
Advanced Network Intrusion Detection System
Uses flow-based feature extraction + CICIDS 2017 trained model
Author: Pratham | EtherWatch IDS v2
"""

import os
import subprocess
import threading
import logging
import time
import pickle
from collections import defaultdict
from datetime import datetime

import requests
from dotenv import load_dotenv
from scapy.all import sniff, IP, TCP, UDP, Raw
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_FILE      = os.getenv("MODEL_FILE", "traffic_model.pkl")
SCALER_FILE     = os.getenv("SCALER_FILE", "scaler.pkl")
ABUSEIPDB_KEY   = os.getenv("ABUSEIPDB_KEY", "")
INTERFACE       = os.getenv("INTERFACE", "eth0")
ENABLE_BLOCKING = os.getenv("ENABLE_BLOCKING", "false").lower() == "true"
FLOW_TIMEOUT    = int(os.getenv("FLOW_TIMEOUT", "60"))

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="alerts.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ─────────────────────────────────────────────
# FLASK + SOCKETIO
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "ids-secret-2024")
socketio = SocketIO(app, cors_allowed_origins="*")

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
BLOCKED_IPS   = set()
THREAT_IPS    = set()
ALERT_HISTORY = []
STATS = {
    "total_packets": 0,
    "total_flows": 0,
    "threats_detected": 0,
    "blocked_ips": 0
}

flows = defaultdict(lambda: {
    "start_time": None,
    "last_seen": None,
    "fwd_packets": [],
    "bwd_packets": [],
    "fwd_flags": [],
    "bwd_flags": [],
    "fwd_header_len": 0,
    "bwd_header_len": 0,
    "init_win_fwd": None,
    "init_win_bwd": None,
})

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
def load_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            m = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            s = pickle.load(f)
        logging.info(f"[MODEL] Loaded {MODEL_FILE} + {SCALER_FILE}")
        return m, s
    except FileNotFoundError:
        logging.error("[MODEL] No model found. Run train.py first!")
        return None, None

model, scaler = load_model()

# ─────────────────────────────────────────────
# THREAT INTELLIGENCE — AbuseIPDB
# ─────────────────────────────────────────────
def check_abuseipdb(ip: str) -> bool:
    if not ABUSEIPDB_KEY:
        return False
    try:
        resp = requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": 90},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data.get("abuseConfidenceScore", 0) >= 50
    except Exception as e:
        logging.warning(f"[ABUSEIPDB] Error checking {ip}: {e}")
    return False

def fetch_threat_intelligence():
    global THREAT_IPS
    if not ABUSEIPDB_KEY:
        logging.warning("[THREAT INTEL] No AbuseIPDB key. Skipping.")
        return
    try:
        resp = requests.get(
            "https://api.abuseipdb.com/api/v2/blacklist",
            headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"},
            params={"confidenceMinimum": 90},
            timeout=10
        )
        if resp.status_code == 200:
            entries = resp.json().get("data", [])
            THREAT_IPS = {e["ipAddress"] for e in entries}
            logging.info(f"[THREAT INTEL] Loaded {len(THREAT_IPS)} malicious IPs")
    except Exception as e:
        logging.error(f"[THREAT INTEL] Fetch error: {e}")

def threat_intel_refresher():
    while True:
        fetch_threat_intelligence()
        time.sleep(3600)

# ─────────────────────────────────────────────
# IP BLOCKING — iptables
# ─────────────────────────────────────────────
def block_ip(ip: str):
    if not ENABLE_BLOCKING:
        logging.info(f"[BLOCK] Blocking disabled. Would block: {ip}")
        return
    if ip in BLOCKED_IPS:
        return
    try:
        subprocess.run(["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                       check=True, capture_output=True)
        subprocess.run(["iptables", "-A", "OUTPUT", "-d", ip, "-j", "DROP"],
                       check=True, capture_output=True)
        BLOCKED_IPS.add(ip)
        STATS["blocked_ips"] = len(BLOCKED_IPS)
        logging.warning(f"[BLOCK] Blocked IP via iptables: {ip}")
    except subprocess.CalledProcessError as e:
        logging.error(f"[BLOCK] iptables failed for {ip}: {e.stderr.decode()}")
    except PermissionError:
        logging.error("[BLOCK] Need root privileges for iptables!")

def unblock_ip(ip: str):
    if ip not in BLOCKED_IPS:
        return False
    try:
        subprocess.run(["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], check=True)
        subprocess.run(["iptables", "-D", "OUTPUT", "-d", ip, "-j", "DROP"], check=True)
        BLOCKED_IPS.discard(ip)
        STATS["blocked_ips"] = len(BLOCKED_IPS)
        logging.info(f"[UNBLOCK] Unblocked IP: {ip}")
        return True
    except Exception as e:
        logging.error(f"[UNBLOCK] Error: {e}")
        return False

# ─────────────────────────────────────────────
# ALERT HANDLER
# ─────────────────────────────────────────────
def raise_alert(level: str, message: str, src_ip: str = None, action: str = "log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "src_ip": src_ip,
        "action": action
    }
    ALERT_HISTORY.append(alert)
    if len(ALERT_HISTORY) > 500:
        ALERT_HISTORY.pop(0)
    logging.warning(f"[{level}] {message}")
    socketio.emit("new_alert", alert)
    STATS["threats_detected"] += 1
    if src_ip and action == "block":
        block_ip(src_ip)

# ─────────────────────────────────────────────
# FLOW FEATURE EXTRACTION
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "flow_duration", "total_fwd_pkts", "total_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_bytes_per_s", "flow_pkts_per_s",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt",
    "down_up_ratio", "avg_pkt_size",
    "fwd_header_len", "bwd_header_len",
    "init_win_bytes_fwd", "init_win_bytes_bwd",
    "active_min", "active_max", "active_mean",
    "idle_min", "idle_max", "idle_mean",
]

def safe_stats(lst):
    if not lst:
        return 0, 0, 0, 0
    arr = np.array(lst)
    return float(arr.max()), float(arr.min()), float(arr.mean()), float(arr.std())

def iat_list(timestamps):
    if len(timestamps) < 2:
        return [0]
    return [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

def extract_flow_features(flow_data: dict) -> dict:
    now   = time.time()
    start = flow_data["start_time"] or now
    last  = flow_data["last_seen"] or now

    fwd_sizes = [p[0] for p in flow_data["fwd_packets"]]
    bwd_sizes = [p[0] for p in flow_data["bwd_packets"]]
    fwd_times = [p[1] for p in flow_data["fwd_packets"]]
    bwd_times = [p[1] for p in flow_data["bwd_packets"]]
    all_sizes = fwd_sizes + bwd_sizes
    duration  = max(last - start, 1e-6)

    fwd_mx, fwd_mn, fwd_me, fwd_sd = safe_stats(fwd_sizes)
    bwd_mx, bwd_mn, bwd_me, bwd_sd = safe_stats(bwd_sizes)
    all_mx, all_mn, all_me, all_sd = safe_stats(all_sizes)

    fwd_iats = iat_list(sorted(fwd_times))
    bwd_iats = iat_list(sorted(bwd_times))
    fi_mx, fi_mn, fi_me, fi_sd = safe_stats(fwd_iats)
    bi_mx, bi_mn, bi_me, bi_sd = safe_stats(bwd_iats)

    total_bytes = sum(all_sizes)
    total_pkts  = len(all_sizes)

    all_flags = flow_data["fwd_flags"] + flow_data["bwd_flags"]
    def flag_count(f): return sum(1 for fl in all_flags if fl & f)

    all_iats = sorted(fwd_iats + bwd_iats)
    active = [i for i in all_iats if i < 1.0]
    idle   = [i for i in all_iats if i >= 1.0]
    a_mx, a_mn, a_me, _ = safe_stats(active)
    i_mx, i_mn, i_me, _ = safe_stats(idle)

    return {
        "flow_duration":      duration,
        "total_fwd_pkts":     len(fwd_sizes),
        "total_bwd_pkts":     len(bwd_sizes),
        "fwd_pkt_len_max":    fwd_mx,
        "fwd_pkt_len_min":    fwd_mn,
        "fwd_pkt_len_mean":   fwd_me,
        "fwd_pkt_len_std":    fwd_sd,
        "bwd_pkt_len_max":    bwd_mx,
        "bwd_pkt_len_min":    bwd_mn,
        "bwd_pkt_len_mean":   bwd_me,
        "bwd_pkt_len_std":    bwd_sd,
        "flow_bytes_per_s":   total_bytes / duration,
        "flow_pkts_per_s":    total_pkts / duration,
        "fwd_iat_mean":       fi_me,
        "fwd_iat_std":        fi_sd,
        "fwd_iat_max":        fi_mx,
        "fwd_iat_min":        fi_mn,
        "bwd_iat_mean":       bi_me,
        "bwd_iat_std":        bi_sd,
        "bwd_iat_max":        bi_mx,
        "bwd_iat_min":        bi_mn,
        "pkt_len_min":        all_mn,
        "pkt_len_max":        all_mx,
        "pkt_len_mean":       all_me,
        "pkt_len_std":        all_sd,
        "pkt_len_var":        all_sd ** 2,
        "fin_flag_cnt":       flag_count(0x01),
        "syn_flag_cnt":       flag_count(0x02),
        "rst_flag_cnt":       flag_count(0x04),
        "psh_flag_cnt":       flag_count(0x08),
        "ack_flag_cnt":       flag_count(0x10),
        "urg_flag_cnt":       flag_count(0x20),
        "down_up_ratio":      len(bwd_sizes) / max(len(fwd_sizes), 1),
        "avg_pkt_size":       total_bytes / max(total_pkts, 1),
        "fwd_header_len":     flow_data["fwd_header_len"],
        "bwd_header_len":     flow_data["bwd_header_len"],
        "init_win_bytes_fwd": flow_data["init_win_fwd"] or 0,
        "init_win_bytes_bwd": flow_data["init_win_bwd"] or 0,
        "active_min":         a_mn,
        "active_max":         a_mx,
        "active_mean":        a_me,
        "idle_min":           i_mn,
        "idle_max":           i_mx,
        "idle_mean":          i_me,
    }

# ─────────────────────────────────────────────
# PACKET PROCESSING
# ─────────────────────────────────────────────
flows_lock = threading.Lock()

def process_packet(packet):
    global flows
    STATS["total_packets"] += 1
    if not packet.haslayer(IP):
        return

    ip_layer = packet[IP]
    src_ip   = ip_layer.src
    dst_ip   = ip_layer.dst
    proto    = ip_layer.proto
    pkt_size = len(packet)
    now      = time.time()

    src_port, dst_port, flags, header_len = 0, 0, 0, 0
    if packet.haslayer(TCP):
        tcp        = packet[TCP]
        src_port   = tcp.sport
        dst_port   = tcp.dport
        flags      = tcp.flags
        header_len = tcp.dataofs * 4
    elif packet.haslayer(UDP):
        udp        = packet[UDP]
        src_port   = udp.sport
        dst_port   = udp.dport
        header_len = 8

    fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
    rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

    with flows_lock:
        if fwd_key in flows:
            key, direction = fwd_key, "fwd"
        elif rev_key in flows:
            key, direction = rev_key, "bwd"
        else:
            key, direction = fwd_key, "fwd"

        flow = flows[key]
        if flow["start_time"] is None:
            flow["start_time"] = now
        flow["last_seen"] = now

        if direction == "fwd":
            flow["fwd_packets"].append((pkt_size, now))
            flow["fwd_flags"].append(flags)
            flow["fwd_header_len"] += header_len
            if flow["init_win_fwd"] is None and packet.haslayer(TCP):
                flow["init_win_fwd"] = packet[TCP].window
        else:
            flow["bwd_packets"].append((pkt_size, now))
            flow["bwd_flags"].append(flags)
            flow["bwd_header_len"] += header_len
            if flow["init_win_bwd"] is None and packet.haslayer(TCP):
                flow["init_win_bwd"] = packet[TCP].window

    if src_ip in THREAT_IPS:
        raise_alert("CRITICAL", f"Known malicious IP: {src_ip} → {dst_ip}", src_ip, "block")
        return

    if src_ip not in BLOCKED_IPS and ABUSEIPDB_KEY:
        threading.Thread(target=_async_abuseipdb_check,
                         args=(src_ip, dst_ip), daemon=True).start()

    with flows_lock:
        f = flows[key]
        syn_pkts = sum(1 for fl in f["fwd_flags"] if fl & 0x02 and not (fl & 0x10))
        if syn_pkts > 50:
            raise_alert("HIGH", f"Possible SYN flood from {src_ip}", src_ip, "block")

    should_analyze = False
    with flows_lock:
        f = flows[key]
        total_fwd = len(f["fwd_packets"])
        fin_rst   = any(fl & 0x01 or fl & 0x04 for fl in f["fwd_flags"])
        if total_fwd >= 20 or (fin_rst and total_fwd >= 5):
            should_analyze = True

    if should_analyze and model and scaler:
        with flows_lock:
            features = extract_flow_features(flows[key])
            flows[key] = flows.default_factory()
        STATS["total_flows"] += 1
        _run_ml_prediction(features, src_ip, dst_ip)

def _async_abuseipdb_check(src_ip, dst_ip):
    if check_abuseipdb(src_ip):
        THREAT_IPS.add(src_ip)
        raise_alert("HIGH", f"AbuseIPDB flagged {src_ip}", src_ip, "block")

def _run_ml_prediction(features: dict, src_ip: str, dst_ip: str):
    try:
        df        = pd.DataFrame([features])[FEATURE_COLUMNS]
        df_scaled = scaler.transform(df)
        pred      = model.predict(df_scaled)[0]
        proba     = model.predict_proba(df_scaled)[0]
        confidence = float(max(proba)) * 100
        if pred != 0:
            label = _cicids_label(pred)
            raise_alert(
                "HIGH",
                f"ML detected {label} from {src_ip} → {dst_ip} (confidence: {confidence:.1f}%)",
                src_ip, "block"
            )
    except Exception as e:
        logging.error(f"[ML] Prediction error: {e}")

def _cicids_label(pred_class: int) -> str:
    labels = {
        1: "DDoS", 2: "PortScan", 3: "BruteForce",
        4: "Infiltration", 5: "BotNet", 6: "WebAttack",
        7: "Heartbleed", 8: "DoS"
    }
    return labels.get(pred_class, f"Attack-{pred_class}")

# ─────────────────────────────────────────────
# FLOW CLEANUP
# ─────────────────────────────────────────────
def flow_cleanup_worker():
    while True:
        time.sleep(30)
        now = time.time()
        with flows_lock:
            expired = [k for k, v in flows.items()
                       if v["last_seen"] and (now - v["last_seen"]) > FLOW_TIMEOUT]
            for k in expired:
                f = flows[k]
                if len(f["fwd_packets"]) >= 5 and model and scaler:
                    features = extract_flow_features(f)
                    threading.Thread(target=_run_ml_prediction,
                                     args=(features, k[0], k[1]), daemon=True).start()
                del flows[k]

# ─────────────────────────────────────────────
# SNIFFING
# ─────────────────────────────────────────────
def start_sniffing():
    logging.info(f"[SNIFF] Starting on interface: {INTERFACE}")
    sniff(iface=INTERFACE, prn=process_packet, store=False, filter="ip")

# ─────────────────────────────────────────────
# FLASK DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>EtherWatch IDS v2</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<style>
  :root {
    --bg:#0a0e1a; --card:#111827; --border:#1e293b;
    --green:#10b981; --red:#ef4444; --yellow:#f59e0b;
    --blue:#3b82f6; --text:#e2e8f0; --muted:#64748b;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Courier New',monospace; }

  header {
    background:var(--card); border-bottom:1px solid var(--border);
    padding:16px 24px; display:flex; align-items:center; gap:12px;
  }
  header h1 { font-size:20px; color:var(--green); letter-spacing:2px; }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--green);
    animation:pulse 2s infinite; }
  @keyframes pulse {
    0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.4);}
    50%{box-shadow:0 0 0 6px rgba(16,185,129,0);}
  }

  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; padding:24px; }
  .stat { background:var(--card); border:1px solid var(--border);
    border-radius:8px; padding:20px; }
  .stat .lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
  .stat .val { font-size:32px; font-weight:700; margin-top:8px; }
  .g .val{color:var(--green);} .r .val{color:var(--red);}
  .y .val{color:var(--yellow);} .b .val{color:var(--blue);}

  .bottom { display:grid; grid-template-columns:2fr 1fr; gap:16px; padding:0 24px 24px; }
  .panel { background:var(--card); border:1px solid var(--border); border-radius:8px; overflow:hidden; }
  .ph {
    padding:12px 16px; border-bottom:1px solid var(--border);
    font-size:12px; text-transform:uppercase; letter-spacing:1px;
    color:var(--muted); display:flex; justify-content:space-between; align-items:center;
  }
  .badge { background:var(--red); color:#fff; border-radius:12px; padding:2px 8px; font-size:11px; }

  #alert-list { height:380px; overflow-y:auto; padding:8px; }
  .ai {
    padding:10px 12px; border-radius:6px; margin-bottom:6px;
    border-left:3px solid var(--muted); font-size:12px;
    animation:fadeIn .3s ease;
  }
  @keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
  .ai.CRITICAL{border-color:var(--red);   background:rgba(239,68,68,.08);}
  .ai.HIGH    {border-color:var(--yellow);background:rgba(245,158,11,.08);}
  .ai.INFO    {border-color:var(--blue);  background:rgba(59,130,246,.08);}
  .ts{color:var(--muted);font-size:10px;}
  .msg{margin-top:3px;}
  .tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:700;margin-right:6px;}
  .tag.CRITICAL{background:var(--red);color:#fff;}
  .tag.HIGH{background:var(--yellow);color:#000;}
  .tag.INFO{background:var(--blue);color:#fff;}

  #blocked-list{height:380px;overflow-y:auto;padding:8px;}
  .bi{
    display:flex;justify-content:space-between;align-items:center;
    padding:8px 10px;border-radius:6px;margin-bottom:4px;
    background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);font-size:12px;
  }
  .ubtn{
    background:none;border:1px solid var(--red);color:var(--red);
    border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px;font-family:'Courier New',monospace;
  }
  .ubtn:hover{background:var(--red);color:#fff;}
  ::-webkit-scrollbar{width:4px;}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
  .empty{color:var(--muted);text-align:center;margin-top:40px;font-size:13px;}
</style>
</head>
<body>

<header>
  <div class="dot"></div>
  <h1>⚡ ETHERWATCH IDS v2</h1>
  <span style="margin-left:auto;font-size:12px;color:var(--muted)" id="uptime">00:00:00</span>
</header>

<div class="grid">
  <div class="stat g"><div class="lbl">Total Packets</div><div class="val" id="tp">0</div></div>
  <div class="stat b"><div class="lbl">Flows Analyzed</div><div class="val" id="tf">0</div></div>
  <div class="stat y"><div class="lbl">Threats Detected</div><div class="val" id="td">0</div></div>
  <div class="stat r"><div class="lbl">Blocked IPs</div><div class="val" id="bi">0</div></div>
</div>

<div class="bottom">
  <div class="panel">
    <div class="ph">Live Alerts <span class="badge" id="abadge">0</span></div>
    <div id="alert-list"><div class="empty">Waiting for alerts...</div></div>
  </div>
  <div class="panel">
    <div class="ph">Blocked IPs</div>
    <div id="blocked-list"><div class="empty">No IPs blocked</div></div>
  </div>
</div>

<script>
  const socket = io();
  let alertCount = 0;
  const t0 = Date.now();

  setInterval(() => {
    const s = Math.floor((Date.now()-t0)/1000);
    const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
    document.getElementById('uptime').textContent =
      `Uptime: ${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  }, 1000);

  socket.on('stats_update', d => {
    document.getElementById('tp').textContent = d.total_packets.toLocaleString();
    document.getElementById('tf').textContent = d.total_flows.toLocaleString();
    document.getElementById('td').textContent = d.threats_detected.toLocaleString();
    document.getElementById('bi').textContent = d.blocked_ips.toLocaleString();
  });

  socket.on('new_alert', a => {
    alertCount++;
    document.getElementById('abadge').textContent = alertCount;
    const list = document.getElementById('alert-list');
    const ph = list.querySelector('.empty');
    if (ph) ph.remove();
    const el = document.createElement('div');
    el.className = `ai ${a.level}`;
    el.innerHTML = `<div class="ts">${a.timestamp}</div>
      <div class="msg"><span class="tag ${a.level}">${a.level}</span>${a.message}</div>`;
    list.prepend(el);
    while (list.children.length > 100) list.removeChild(list.lastChild);
    if (a.action === 'block' && a.src_ip) addBlocked(a.src_ip);
  });

  function addBlocked(ip) {
    const list = document.getElementById('blocked-list');
    const ph = list.querySelector('.empty');
    if (ph) ph.remove();
    const id = 'b-' + ip.replace(/\./g,'-');
    if (document.getElementById(id)) return;
    const el = document.createElement('div');
    el.className = 'bi'; el.id = id;
    el.innerHTML = `<span>${ip}</span>
      <button class="ubtn" onclick="unblock('${ip}')">UNBLOCK</button>`;
    list.prepend(el);
  }

  function unblock(ip) {
    fetch('/unblock/'+ip, {method:'POST'}).then(r=>r.json()).then(d => {
      if (d.success) {
        const el = document.getElementById('b-'+ip.replace(/\./g,'-'));
        if (el) el.remove();
      }
    });
  }

  fetch('/alerts').then(r=>r.json()).then(alerts => alerts.forEach(a => socket.emit('new_alert', a)));
</script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/alerts")
def get_alerts():
    return jsonify(ALERT_HISTORY[-100:])

@app.route("/stats")
def get_stats():
    return jsonify(STATS)

@app.route("/blocked")
def get_blocked():
    return jsonify(list(BLOCKED_IPS))

@app.route("/unblock/<ip>", methods=["POST"])
def unblock(ip):
    success = unblock_ip(ip)
    return jsonify({"success": success, "ip": ip})

@app.route("/flows")
def get_active_flows():
    with flows_lock:
        active = {str(k): len(v["fwd_packets"]) + len(v["bwd_packets"])
                  for k, v in flows.items()}
    return jsonify({"active_flows": len(active), "flows": active})

@socketio.on("connect")
def on_connect():
    emit("stats_update", STATS)

def stats_broadcaster():
    while True:
        time.sleep(5)
        socketio.emit("stats_update", STATS)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("  EtherWatch IDS v2 — Starting")
    logging.info(f"  Interface    : {INTERFACE}")
    logging.info(f"  IP Blocking  : {'ENABLED' if ENABLE_BLOCKING else 'DISABLED'}")
    logging.info(f"  Model Loaded : {model is not None}")
    logging.info("=" * 60)

    threads = [
        threading.Thread(target=threat_intel_refresher, daemon=True),
        threading.Thread(target=start_sniffing,         daemon=True),
        threading.Thread(target=flow_cleanup_worker,    daemon=True),
        threading.Thread(target=stats_broadcaster,      daemon=True),
    ]
    for t in threads:
        t.start()

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
