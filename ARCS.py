import os
import sys
import math
import json
import io
import tempfile
import base64
import pickle
import sqlite3
from datetime import timedelta, datetime
import time
import warnings
import random

# ==========================================
# 1. IMPORT LIBRARY 
# (Pastikan Anda menggunakan requirements.txt di GitHub)
# ==========================================
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
)
from tensorflow.keras.callbacks import EarlyStopping
from fpdf import FPDF
from PIL import Image

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="ARCS | Aircraft Reliability Control Systems", layout="wide", page_icon="✈️", initial_sidebar_state="expanded")

# --- INISIALISASI SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'employee_name' not in st.session_state:
    st.session_state['employee_name'] = ""
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'engines' not in st.session_state:
    st.session_state['engines'] = []
if 'run_time' not in st.session_state:
    st.session_state['run_time'] = None
if 'analyzer_name' not in st.session_state:
    st.session_state['analyzer_name'] = None
if 'thesis_metrics' not in st.session_state:
    st.session_state['thesis_metrics'] = None
if 'current_engine' not in st.session_state:
    st.session_state['current_engine'] = None

# --- PERBAIKAN: BULLETPROOF SEED LOCKING ---
def reset_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    tf.keras.backend.clear_session()


# ==========================================
# 3. LOGIN DASHBOARD UI
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("""
        <style>
            .stApp { background: linear-gradient(135deg, #1c3c7e 0%, #0a1938 100%); }
            header[data-testid="stHeader"] { background: transparent !important; }
            [data-testid="collapsedControl"], .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
            div[data-testid="column"]:nth-of-type(2) {
                background-color: white; padding: 40px 50px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
            }
            .stTextInput input { border-radius: 6px; padding: 10px 15px; border: 1px solid #ddd; background-color: #f8f9fa; color: #333;}
            div.stButton > button {
                width: 100%; background-color: #2563eb; color: white; font-weight: bold; border-radius: 6px; padding: 10px; border: none; margin-top: 10px;
            }
            div.stButton > button:hover { background-color: #1d4ed8; color: white; }
            .footer-login {
                position: fixed; bottom: 20px; width: 100%; text-align: center; color: rgba(255,255,255,0.7); font-size: 12px; left: 0;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 40px; font-weight: 900; color: #000; letter-spacing: 1px; line-height: 1;">ARCS</h1>
                <div style="width: 2px; height: 35px; background-color: #000; margin: 0 15px;"></div>
                <div style="text-align: left; line-height: 1.1;">
                    <span style="display: block; font-weight: 800; font-size: 15px; color: #000;">Aircraft Reliability</span>
                    <span style="display: block; font-weight: 800; font-size: 15px; color: #000;">Control Systems</span>
                </div>
            </div>
            <h3 style="font-size: 16px; margin-bottom: 5px; color: #000;">Login</h3>
        """, unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
        password = st.text_input("Password", placeholder="Password", type="password", label_visibility="collapsed")
        if st.button("Log In"):
            if password == "TEA2" and username.strip():
                st.session_state['logged_in'] = True
                st.session_state['employee_name'] = username.strip()
                st.rerun()
            else:
                st.error("Invalid Username or Password.")
        st.markdown('<div style="text-align:center; font-size: 11px; color: #888; margin-top:10px; margin-bottom: 10px;">Need help logging in? <a href="#" style="color:#2563eb; text-decoration:none;">Contact the ARCS Help Desk</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-login">Copyright ©2026 Engineering Services GMF AeroAsia In Collaboration with Diponegoro University. All rights reserved.</div>', unsafe_allow_html=True)
    st.stop()


# ==========================================
# 4. SIDEBAR & NAVIGATION ROUTING
# ==========================================
st.sidebar.markdown("### ✈️ Fleet Navigation")
nav_engine = st.sidebar.selectbox("Engine Model", ["GE90-115B", "CFM56-5B"])

# ==========================================
# PERSISTENT APP MEMORY
# ==========================================
# SQLite dipakai agar hasil analisis dapat dibaca ulang saat browser di-refresh
# atau saat aplikasi dibuka dari device lain selama Streamlit Cloud container masih aktif.
# Perhitungan forecast tidak diubah; yang disimpan adalah hasil akhir setelah analisis selesai.
SESSION_DB_FILE = os.environ.get("ARCS_SESSION_DB", "arcs_memory.sqlite3")
LEGACY_SESSION_FILE = f"latest_session_{nav_engine}.pkl"

def init_session_db():
    with sqlite3.connect(SESSION_DB_FILE, timeout=30) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                engine_model TEXT PRIMARY KEY,
                run_time TEXT,
                analyzer_name TEXT,
                engines_blob BLOB,
                results_blob BLOB,
                thesis_metrics_blob BLOB,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

def save_analysis_session(engine_model, session_payload):
    init_session_db()
    with sqlite3.connect(SESSION_DB_FILE, timeout=30) as conn:
        conn.execute(
            """
            INSERT INTO analysis_sessions (
                engine_model, run_time, analyzer_name, engines_blob,
                results_blob, thesis_metrics_blob, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_model) DO UPDATE SET
                run_time=excluded.run_time,
                analyzer_name=excluded.analyzer_name,
                engines_blob=excluded.engines_blob,
                results_blob=excluded.results_blob,
                thesis_metrics_blob=excluded.thesis_metrics_blob,
                updated_at=excluded.updated_at
            """,
            (
                engine_model,
                session_payload.get("run_time"),
                session_payload.get("analyzer_name"),
                sqlite3.Binary(pickle.dumps(session_payload.get("engines"), protocol=pickle.HIGHEST_PROTOCOL)),
                sqlite3.Binary(pickle.dumps(session_payload.get("results"), protocol=pickle.HIGHEST_PROTOCOL)),
                sqlite3.Binary(pickle.dumps(session_payload.get("thesis_metrics"), protocol=pickle.HIGHEST_PROTOCOL)),
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        conn.commit()

def load_analysis_session(engine_model):
    init_session_db()
    with sqlite3.connect(SESSION_DB_FILE, timeout=30) as conn:
        row = conn.execute(
            """
            SELECT run_time, analyzer_name, engines_blob, results_blob, thesis_metrics_blob
            FROM analysis_sessions
            WHERE engine_model = ?
            """,
            (engine_model,)
        ).fetchone()

    if not row:
        return None

    run_time, analyzer_name, engines_blob, results_blob, thesis_metrics_blob = row
    return {
        "run_time": run_time,
        "analyzer_name": analyzer_name,
        "engines": pickle.loads(engines_blob) if engines_blob else [],
        "results": pickle.loads(results_blob) if results_blob else None,
        "thesis_metrics": pickle.loads(thesis_metrics_blob) if thesis_metrics_blob else None,
    }

def load_legacy_pickle_session():
    if not os.path.exists(LEGACY_SESSION_FILE):
        return None
    try:
        with open(LEGACY_SESSION_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def hydrate_session_state_from_memory(engine_model):
    if st.session_state.get('results') is not None:
        return

    saved_session = None
    try:
        saved_session = load_analysis_session(engine_model)
    except Exception as e:
        st.warning(f"Gagal membaca database memori ARCS: {e}")

    # Backward compatibility: jika sebelumnya masih ada file .pkl lama, migrasikan ke SQLite.
    if saved_session is None:
        saved_session = load_legacy_pickle_session()
        if saved_session is not None:
            try:
                save_analysis_session(engine_model, saved_session)
            except Exception:
                pass

    if saved_session is not None:
        st.session_state['run_time']       = saved_session.get('run_time')
        st.session_state['analyzer_name']  = saved_session.get('analyzer_name')
        st.session_state['engines']        = saved_session.get('engines')
        st.session_state['results']        = saved_session.get('results')
        st.session_state['thesis_metrics'] = saved_session.get('thesis_metrics')

if st.session_state.get('current_engine') != nav_engine:
    st.session_state['current_engine'] = nav_engine
    st.session_state['results'] = None
    st.session_state['engines'] = []
    st.session_state['thesis_metrics'] = None

hydrate_session_state_from_memory(nav_engine)

if nav_engine == "GE90-115B":
    nav_module = st.sidebar.radio("Module", ["Home", "Fuel Filter Replacement Forecasting", "Engine Health Analytics"])
else:
    nav_module = st.sidebar.radio("Module", ["Home", "Engine Health Analytics"])

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Log Out", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()


# ==========================================
# 5. CSS & HEADER DASHBOARD UTAMA
# ==========================================
def get_base64_image(file_path):
    try:
        with open(file_path, "rb") as f: return base64.b64encode(f.read()).decode()
    except Exception: return None

te_logo_b64 = get_base64_image("TE.png")
if te_logo_b64:
    logo_html = f'<img src="data:image/png;base64,{te_logo_b64}" style="height: 45px; margin-top: 2px;">'
else:
    logo_html = '<div style="color:#002561; font-weight:bold; font-size:20px;">Engineering Services</div>'

user_display_name = st.session_state['employee_name']

st.markdown(f"""
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f4f4; color: #333; }}
        .stApp {{ background: #f4f4f4 !important; background-color: #f4f4f4 !important; }}
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        .stDeployButton {{ display: none !important; }}
        .block-container {{ padding-top: 3.5rem !important; }}
        
        .garuda-header {{
            background-color: #ffffff; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
            padding: 15px 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-bottom: 2px solid #005eb8;
            border-radius: 8px; margin-bottom: 20px; gap: 15px; padding-left: 60px;
        }}
        .header-left {{ display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }}
        .header-title {{ color: #002561; font-size: 20px; font-weight: 800; text-transform: uppercase; margin: 0; letter-spacing: 0.5px;}}
        .login-link {{ text-decoration: none; color: #005eb8; font-size: 14px; font-weight: 500; display: flex; align-items: center; }}
        
        .card-tool {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); width: 100%; margin-bottom: 20px; }}
        div.stButton > button {{ padding: 14px; background-color: #005eb8; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }}
        div.stButton > button:hover {{ background-color: #002561; color: white; }}
        .cnr-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 15px; margin-bottom: 20px; }}
        .cnr-table th {{ background-color: #f1f3f5; color: #495057; font-weight: 600; text-align: left; padding: 8px; border-bottom: 2px solid #dee2e6; }}
        .cnr-table td {{ padding: 8px; border-bottom: 1px solid #dee2e6; color: #212529; }}
        .info-box {{ background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 15px; margin-top: 15px; margin-bottom: 15px; font-size: 13px; }}
        .info-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; border-bottom: 1px dashed #ddd; padding-bottom: 3px; }}
        .info-label {{ font-weight: 600; color: #555; }}
        .info-val {{ color: #000; font-weight: 500; }}
        .streamlit-expanderHeader {{ font-weight: bold; color: #002561; background-color: #e9ecef; border-radius: 5px; }}
        .thesis-section {{ padding: 10px; }}
        
        .top3-box {{
            background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 20px; border-radius: 5px; margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .top3-header {{ color: #856404; margin: 0 0 10px 0; font-size: 18px; font-weight: bold; }}
        .top3-item {{ margin-bottom: 15px; border-bottom: 1px dashed #e5c77b; padding-bottom: 10px; }}
        .top3-item:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
        .top3-esn {{ font-size: 16px; font-weight: 800; color: #002561; }}
        .top3-reg {{ font-size: 14px; color: #555; }}
        .top3-psi {{ font-size: 18px; font-weight: bold; color: #dc3545; float: right; }}
        .top3-dates {{ font-size: 13px; color: #333; margin-top: 5px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;}}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="garuda-header">
        <div class="header-left">
            <div style="padding-right:15px; border-right: 2px solid #ddd; display: flex; align-items: center;">
                {logo_html}
            </div>
            <h1 class="header-title">Aircraft Reliability Control Systems</h1>
        </div>
        <div class="header-right"><a href="#" class="login-link">Hello, {user_display_name}!</a></div>
    </div>
""", unsafe_allow_html=True)

def color_status(val):
    if   "CRITICAL"     in val: return 'background-color: #dc3545; color: white; font-weight: bold;'
    elif "WARNING"      in val: return 'background-color: #ffc107; color: white; font-weight: bold;'
    elif "ON WATCH"     in val: return 'background-color: #006400; color: white; font-weight: bold;'
    elif "PARKED"       in val: return 'background-color: #6c757d; color: white; font-weight: bold;'
    elif "INSUFFICIENT" in val: return 'background-color: #17a2b8; color: white; font-weight: bold;'
    else:                       return 'background-color: #28a745; color: white; font-weight: bold;'


# ==========================================
# 6. ROUTING HALAMAN
# ==========================================

if nav_module == "Engine Health Analytics":
    st.markdown(f"<br><br><br><h1 style='text-align: center; color: #555;'>🚧 Coming Soon</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: #888;'>The {nav_module} module for {nav_engine} is currently under development.</h4>", unsafe_allow_html=True)
    st.stop()

# ----------------- HALAMAN HOME -----------------
elif nav_module == "Home":
    st.markdown('<div class="card-tool">', unsafe_allow_html=True)
    
    clock_html = """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; font-weight: 600; color: #002561; padding: 10px 15px; background: #e9ecef; border-radius: 6px; border-left: 4px solid #005eb8; display: inline-block; margin-bottom: 15px;">
        🕒 Current Time (UTC): <span id="clock" style="color: #000; font-weight: bold; font-family: monospace; font-size: 16px;"></span>
    </div>
    <script>
        function updateTime() {
            const now = new Date();
            document.getElementById('clock').textContent = now.toUTCString();
        }
        setInterval(updateTime, 1000);
        updateTime();
    </script>
    """
    components.html(clock_html, height=60)
    
    if os.path.exists("poster.png"):
        try:
            st.image("poster.png", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
        except Exception:
            pass
    
    st.markdown(f'<h2 style="color: #002561; border-bottom: 2px solid #005eb8; margin-top: 0;">Executive Overview: {nav_engine}</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('results') is not None:
        run_time = st.session_state.get("run_time", "-")
        results  = st.session_state.get("results", [])
        
        st.markdown(f"**Last Data Sync:** {run_time} (UTC)")
        
        sorted_res = sorted(results, key=lambda x: float(x['PSI']), reverse=True)
        top_3 = sorted_res[:3]

        if len(top_3) > 0:
            st.markdown("<br><div class='top3-box'>", unsafe_allow_html=True)
            st.markdown("<h3 class='top3-header'>🚨 Top Priority Engines (Highest Delta Pressure)</h3>", unsafe_allow_html=True)
            
            for item in top_3:
                esn = item['ESN']
                reg = item['Reg']
                psi = item['PSI']
                p_date = item['Planner Date']
                d_date = item['Date']
                last_flight = item['Dates Info'].get('Last Flight TO', '-')
                
                st.markdown(f"""
                <div class='top3-item'>
                    <span class='top3-esn'>ESN: {esn}</span> <span class='top3-reg'>({reg})</span>
                    <span class='top3-psi'>{psi} PSID</span>
                    <div class='top3-dates'>
                        <span><b>Last Flight:</b> <span style='color:#005eb8;'>{last_flight}</span></span>
                        <span><b>Planner Action Date:</b> <span style='color:#d39e00;'>{p_date}</span></span>
                        <span><b>Est. Due Date:</b> <span style='color:#dc3545;'>{d_date}</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("✅ Seluruh armada dalam kondisi optimal. Tidak ada peringatan prioritas saat ini.")

        st.info(f"💡 Untuk melihat analisis menyeluruh, *Fleet Health Summary*, dan mencetak MSAO, silakan masuk ke menu **Fuel Filter Replacement Forecasting** di Sidebar.")
    else:
        st.info(f"ℹ️ Belum ada data analisis yang tersimpan untuk armada {nav_engine}. Silakan jalankan analisis di modul terkait.")
        
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- HALAMAN FORECASTING -----------------
elif nav_module == "Fuel Filter Replacement Forecasting":

    def build_sequences_strided(arr2d: np.ndarray, window_size: int) -> np.ndarray:
        N, F = arr2d.shape
        out_len = N - window_size
        if out_len <= 0: return np.empty((0, window_size, F), dtype=arr2d.dtype)
        shape   = (out_len, window_size, F)
        strides = (arr2d.strides[0], arr2d.strides[0], arr2d.strides[1])
        view    = np.lib.stride_tricks.as_strided(arr2d, shape=shape, strides=strides)
        return view.copy()  

    def vectorized_monte_carlo(curr_psi: float, base_curve: np.ndarray, n_iter: int, n_steps: int) -> np.ndarray:
        base_arr     = base_curve[:n_steps]
        base_delta   = base_arr - base_arr[0]                       
        start_noises = np.random.normal(0.0,  0.05, n_iter)         
        drift_rfs    = np.random.normal(1.0,  0.15, n_iter)         
        return (curr_psi + start_noises[:, None]) + drift_rfs[:, None] * base_delta[None, :]

    class PDFReport(FPDF):
        def __init__(self):
            super().__init__()
            self.utc_now = datetime.utcnow().strftime('%d %b %Y %H:%M:%S')

        @staticmethod
        def _safe_pdf_text(value):
            """
            PyFPDF/fpdf==1.7.2 memakai encoding Latin-1.
            Karakter di luar Latin-1, misalnya emoji atau smart quotes,
            bisa membuat proses output PDF gagal atau file hasil download rusak.
            """
            if value is None:
                return ""
            return str(value).encode("latin-1", "replace").decode("latin-1")

        def cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
            return super().cell(w, h, self._safe_pdf_text(txt), border, ln, align, fill, link)

        def write(self, h, txt='', link=''):
            return super().write(h, self._safe_pdf_text(txt), link)

        def multi_cell(self, w, h, txt='', border=0, align='J', fill=False):
            return super().multi_cell(w, h, self._safe_pdf_text(txt), border, align, fill)

        def footer(self):
            self.set_y(-15)
            self.set_font("Courier", 'I', 7)
            self.set_text_color(153, 153, 153)
            self.cell(100, 10, f"ARCS Dashboard Generated on {self.utc_now}Z", align='L')
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='R')

    def generate_cnr_pdf(res, user_name="[Nama]", user_phone="[Nomor Telepon]", user_email="[Email]", images_bytes_list=None, notes=None):
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        pdf.set_font("Courier", 'B', 16)
        pdf.cell(0, 10, "Pre-Info Notification", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Courier", 'B', 10)
        pdf.cell(40, 6, "Airline/Customer:"); pdf.set_font("Courier", '', 10); pdf.cell(70, 6, "PT Garuda Indonesia Persero Tbk")
        pdf.set_font("Courier", 'B', 10); pdf.cell(35, 6, "Aircraft Tail:"); pdf.set_font("Courier", '', 10); pdf.cell(45, 6, str(res['Reg']), ln=True)
        pdf.set_font("Courier", 'B', 10)
        pdf.cell(40, 6, "Engine Type:"); pdf.set_font("Courier", '', 10); pdf.cell(70, 6, "GE90-115B")
        pdf.set_font("Courier", 'B', 10); pdf.cell(35, 6, "Engine Serial:"); pdf.set_font("Courier", '', 10); pdf.cell(45, 6, str(res['ESN']), ln=True)
        pdf.set_font("Courier", 'B', 10)
        pdf.cell(40, 6, "Date:"); pdf.set_font("Courier", '', 10); pdf.cell(70, 6, datetime.now().strftime('%d %b %Y'))
        pdf.set_font("Courier", 'B', 10); pdf.cell(35, 6, "Aircraft Type:"); pdf.set_font("Courier", '', 10); pdf.cell(45, 6, "B777-300ER", ln=True)
        
        pdf.ln(5)
        pdf.set_line_width(0.2)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Courier", 'B', 11); pdf.cell(0, 6, "1. Summary", ln=True)
        pdf.set_font("Courier", '', 10)
        pdf.write(5, "Based on engine trend monitoring, Fuel Filter Delta Pressure of ")
        pdf.set_font("Courier", 'B', 10)
        pdf.write(5, str(res['Reg']))
        pdf.set_font("Courier", '', 10)
        pdf.write(5, " with ")
        pdf.set_font("Courier", 'B', 10)
        pdf.write(5, f"ESN {res['ESN']}")
        pdf.set_font("Courier", '', 10)
        pdf.write(5, " has gradually increased over the past three months, with a significant increase in the last few days, reaching ")
        pdf.set_font("Courier", 'B', 10)
        pdf.write(5, f"{res['PSI']} psid")
        pdf.set_font("Courier", '', 10)
        pdf.write(5, " during the takeoff phase.")
        pdf.ln(8)

        pdf.set_font("Courier", 'B', 11); pdf.cell(0, 6, "2. Analytics", ln=True)
        pdf.set_font("Courier", '', 10)
        pdf.write(5, "The analysis results show that there is a possibility that a CNR will appear regarding a contaminated fuel filter on ")
        pdf.set_font("Courier", 'B', 10)
        pdf.write(5, str(res['Date']))
        pdf.set_font("Courier", '', 10)
        pdf.write(5, ".")
        pdf.ln(8)

        pdf.set_font("Courier", 'B', 11); pdf.cell(0, 6, "3. Recommendation", ln=True)
        pdf.set_font("Courier", '', 10)
        
        pdf.cell(0, 5, "Dear MS-MCC,", ln=True)
        pdf.write(5, "Please help us create an MSAO to perform the following tasks during the next maintenance or when the CNR is released:")
        pdf.ln(6)
        pdf.cell(0, 5, "- Fuel Filter Element Removal (AMM TASK 73-11-02-000-801-H01).", ln=True)
        pdf.cell(0, 5, "- Fuel Filter Element Installation (AMM TASK 73-11-02-400-801-H01).", ln=True)
        pdf.ln(4)
        
        pdf.cell(0, 5, "Dear TLP,", ln=True)
        pdf.write(5, f"With this document, it is possible that a CNR will appear on the {res['Date']}. Please to prepare ground time so that the CNR troubleshooting process can be carried out.")
        pdf.ln(8)

        pdf.set_font("Courier", 'B', 11); pdf.cell(0, 6, "4. Supporting Information", ln=True)
        pdf.set_font("Courier", '', 9)
        pdf.cell(55, 5, "Reference Baseline Date:"); pdf.cell(0, 5, res['Dates Info']['Ref Date'], ln=True)
        pdf.cell(55, 5, "Observation Date:");         pdf.cell(0, 5, res['Dates Info']['Obs Date'], ln=True)
        pdf.cell(55, 5, "Last Flight Date (CR/TO):"); pdf.cell(0, 5, f"{res['Dates Info']['Last Flight CR']} / {res['Dates Info']['Last Flight TO']}", ln=True)
        pdf.ln(3)

        pdf.set_font("Courier", 'B', 9)
        pdf.cell(30, 6, "Phase",          border=1, align='C')
        pdf.cell(40, 6, "Value at Ref.",  border=1, align='C')
        pdf.cell(40, 6, "Value at Obs.",  border=1, align='C')
        pdf.cell(40, 6, "Overall Change", border=1, align='C', ln=True)

        pdf.set_font("Courier", '', 9)
        for row in res['CNR Table']:
            pdf.cell(30, 6, str(row['Flight Phase']),       border=1, align='C')
            pdf.cell(40, 6, str(row['Value at Ref. Date']), border=1, align='C')
            pdf.cell(40, 6, str(row['Value at Obs. Date']), border=1, align='C')
            pdf.cell(40, 6, str(row['Overall Change']),     border=1, align='C', ln=True)
        pdf.ln(5)

        if images_bytes_list and len(images_bytes_list) > 0:
            for img_bytes in images_bytes_list:
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode in ('RGBA', 'P', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if len(img.split()) >= 4: background.paste(img, mask=img.split()[3])
                        else: background.paste(img)
                        img = background
                    elif img.mode != 'RGB': img = img.convert('RGB')
                    
                    img_w_px, img_h_px = img.size
                    aspect_ratio = img_h_px / img_w_px
                    pdf_w = 190 
                    pdf_h = pdf_w * aspect_ratio
                    
                    x_pos = 10
                    if pdf_h > 240:
                        pdf_h = 240
                        pdf_w = pdf_h / aspect_ratio
                        x_pos = 10 + (190 - pdf_w) / 2 
                    
                    if pdf.get_y() + pdf_h > 270: pdf.add_page()
                    else: pdf.ln(5)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        img.save(tmp, format='JPEG', quality=95)
                        tmp_path = tmp.name
                    
                    pdf.image(tmp_path, x=x_pos, w=pdf_w, h=pdf_h) 
                    os.unlink(tmp_path)
                    pdf.ln(5)
                except Exception as e:
                    pdf.set_font("Courier", 'I', 9); pdf.set_text_color(255, 0, 0)
                    pdf.cell(0, 5, f"[Error processing uploaded image: {e}]", ln=True)
                    pdf.set_text_color(0, 0, 0); pdf.ln(5)

        if notes and str(notes).strip() != "":
            if pdf.get_y() > 250: pdf.add_page()
            else: pdf.ln(2)
            pdf.set_font("Courier", 'B', 11); pdf.cell(0, 6, "6. Notes", ln=True)
            pdf.set_font("Courier", '', 10)
            pdf.write(5, str(notes).strip())
            pdf.ln(8)

        if pdf.get_y() > 245: pdf.add_page()
        else: pdf.ln(5)
            
        pdf.set_font("Courier", 'B', 10)
        pdf.cell(0, 5, "DISCLAIMER:", ln=True)
        pdf.set_font("Courier", 'I', 9)
        pdf.write(5, "This document provides an estimated forecast and should not be used as the absolute baseline for maintenance execution. Please continue to periodically monitor the engine data through the Engine Health Portal.")
        pdf.ln(8)

        if pdf.get_y() > 220: pdf.add_page()
        else: pdf.ln(5) 
            
        pdf.set_font("Courier", 'I', 10)
        pdf.cell(0, 5, "Best Regards,", ln=True)
        pdf.cell(0, 5, user_name, ln=True)
        pdf.cell(0, 5, "Powerplant Engineering - TEA 2", ln=True)
        pdf.ln(3)

        if os.path.exists("GMF.png"):
            try:
                pdf.image("GMF.png", x=pdf.get_x(), w=40)
                pdf.ln(3)
            except Exception:
                # Jangan gagalkan PDF hanya karena file logo rusak/tidak kompatibel.
                pdf.ln(5)
        else: pdf.ln(5) 

        pdf.set_font("Courier", 'I', 10)
        pdf.cell(0, 5, "PT Garuda Maintenance Facility Aero Asia Tbk", ln=True)
        pdf.cell(0, 5, f"P : {user_phone}", ln=True)
        pdf.cell(0, 5, f"E : {user_email}", ln=True)

        pdf.ln(5)
        pdf.set_text_color(150, 150, 150)
        pdf.set_font("Courier", 'I', 7)
        pdf.write(4, "This message may contain confidential and/or proprietary information of Garuda Maintenance Facility Aero Asia, PT., and/or their affiliated companies.")
        pdf.set_text_color(0, 0, 0) 

        pdf_payload = pdf.output(dest="S")
        if isinstance(pdf_payload, str):
            return pdf_payload.encode("latin-1")
        return bytes(pdf_payload)

    # --- LOGIKA FISIKA & PARAMETER AI ---
    warnings.filterwarnings('ignore')

    CYCLES_PER_DAY         = 2
    HOURS_PER_CYCLE        = 7
    WINDOW_SIZE            = 60
    PREDICT_STEPS          = 500
    CUSTOM_TARGET          = 14.0
    MAX_Y_LIMIT            = 35.0
    TAKEOFF_DAMPING_FACTOR = 0.15
    MIN_DRIFT_RATE         = 0.0001
    MC_ITERATIONS          = 50
    PARETO_CONFIDENCE      = 80
    INITIAL_POROSITY       = 0.55
    MODEL_PATH             = "arcs_gru_model.keras"
    HISTORY_PATH           = "arcs_gru_history.json"
    MIN_DATA_POINTS_30D    = 10
    RAPID_SHIFT_THRESHOLD  = 3.0
    GE_SOFT_LIMIT          = 14.0
    LOW_PRESSURE_SAFEGUARD = 10.0

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self, progress_bar, status_text, total_epochs):
            self.progress_bar = progress_bar
            self.status_text  = status_text
            self.total_epochs = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            pct = (epoch + 1) / self.total_epochs
            self.progress_bar.progress(pct)
            self.status_text.text(f"🧠 Training AI Model... Epoch {epoch + 1}/{self.total_epochs} | Loss: {logs['loss']:.4f}")

    # --- DASHBOARD FORECASTING ---
    with st.container():
        st.markdown('<div class="card-tool">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: #002561; border-bottom: 2px solid #005eb8; margin-top: 0;">Fuel Filter Replacement Forecasting: {nav_engine}</h2>', unsafe_allow_html=True)

        col_up, col_opt = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("Upload Sensor Data (Excel/CSV)", type=['xlsx', 'csv'])
        with col_opt:
            st.markdown("<br>", unsafe_allow_html=True)
            force_retrain = st.checkbox("Force Retrain AI Model", value=False, help="Abaikan memori model yang tersimpan dan latih AI dari nol.")

        if st.button("🚀 Run Analysis", key="btn_run"):
            if uploaded_file is not None:
                reset_seeds(42)
                
                st.session_state['run_time'] = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                st.session_state['analyzer_name'] = st.session_state['employee_name']
                
                start_time   = time.time()
                progress_bar = st.progress(0)
                status_text  = st.empty()

                status_text.text("📂 Reading Data...")

                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, skiprows=7)
                else: df = pd.read_excel(uploaded_file, skiprows=7)

                df = df.iloc[:, [5, 3, 4, 10, 11, 1]].copy()
                df.columns = ['Date', 'ESN', 'Phase', 'Pressure_Raw', 'Pressure_Smoothed_Source', 'Aircraft_ID']

                df['Phase']                    = df['Phase'].astype(str).str.strip().str.upper()
                df['Date']                     = pd.to_datetime(df['Date'], errors='coerce')
                df['ESN']                      = df['ESN'].astype(str).str.replace('.0', '', regex=False)
                df['Pressure_Final']           = pd.to_numeric(df['Pressure_Raw'],              errors='coerce') / 10000.0
                df['Pressure_Smoothed_Final']  = pd.to_numeric(df['Pressure_Smoothed_Source'],  errors='coerce') / 10000.0

                df           = df.dropna(subset=['Pressure_Final', 'ESN', 'Date']).sort_values(['ESN', 'Date'])
                df_takeoff   = df[df['Phase'] == 'TAKEOFF'].copy()
                dataset_latest_date = df['Date'].max()
                progress_bar.progress(10)

                status_text.text("🛠️ Preprocessing & Time-Based Splitting...")
                engines          = sorted(df['ESN'].unique())
                processed_frames = []

                for eng in engines:
                    mask   = df_takeoff['ESN'] == eng
                    df_eng = df_takeoff[mask].copy().reset_index(drop=True)
                    df_eng['Diff'] = df_eng['Pressure_Final'].diff()
                    candidates = df_eng[df_eng['Diff'] < -1.5].index.tolist()
                    valid_indices = []
                    if candidates:
                        last_idx = -999
                        for idx in candidates:
                            if idx - last_idx > 60:
                                valid_indices.append(idx)
                                last_idx = idx

                    df_eng['Filter_Segment'] = 0
                    current_seg, last_cut    = 0, 0
                    for idx in valid_indices:
                        df_eng.loc[last_cut:idx, 'Filter_Segment'] = current_seg
                        current_seg += 1
                        last_cut     = idx + 1
                    df_eng.loc[last_cut:, 'Filter_Segment'] = current_seg
                    df_eng['Filter_Age']     = df_eng.groupby('Filter_Segment').cumcount()
                    processed_frames.append(df_eng)

                df_train_all = pd.concat(processed_frames)
                train_list, val_list = [], []
                for eng in engines:
                    eng_df     = df_train_all[df_train_all['ESN'] == eng].sort_values('Date')
                    split_idx  = int(len(eng_df) * 0.8)
                    train_list.append(eng_df.iloc[:split_idx])
                    val_list.append(eng_df.iloc[split_idx:])

                df_train_set = pd.concat(train_list)
                df_val_set   = pd.concat(val_list)

                scaler_p = MinMaxScaler(feature_range=(0, 1))
                scaler_a = MinMaxScaler(feature_range=(0, 1))
                scaler_p.fit(df_train_set[['Pressure_Final']])
                scaler_a.fit(df_train_set[['Filter_Age']])

                df_train_all['P_Scaled']   = scaler_p.transform(df_train_all[['Pressure_Final']])
                df_train_all['Age_Scaled'] = scaler_a.transform(df_train_all[['Filter_Age']])
                df_train_set['P_Scaled']   = scaler_p.transform(df_train_set[['Pressure_Final']])
                df_train_set['Age_Scaled'] = scaler_a.transform(df_train_set[['Filter_Age']])
                df_val_set['P_Scaled']     = scaler_p.transform(df_val_set[['Pressure_Final']])
                df_val_set['Age_Scaled']   = scaler_a.transform(df_val_set[['Filter_Age']])

                status_text.text("🗂️ Pre-filtering engine DataFrames...")
                df_by_esn         = {eng: df[df['ESN'] == eng].copy()             for eng in engines}
                processed_by_esn  = {eng: df_train_all[df_train_all['ESN'] == eng].copy() for eng in engines}

                df_takeoff_by_esn, df_cruise_by_esn, df_climb_by_esn = {}, {}, {}
                for eng in engines:
                    d = df_by_esn[eng]
                    df_takeoff_by_esn[eng] = d[d['Phase'] == 'TAKEOFF'].sort_values('Date').reset_index(drop=True)
                    df_cruise_by_esn[eng]  = d[d['Phase'] == 'CRUISE' ].sort_values('Date').reset_index(drop=True)
                    df_climb_by_esn[eng]   = d[d['Phase'] == 'CLIMB'  ].sort_values('Date').reset_index(drop=True)

                _phase_lookup = {'CRUISE': df_cruise_by_esn, 'TAKEOFF': df_takeoff_by_esn, 'CLIMB': df_climb_by_esn}

                X_val_parts, y_val_parts = [], []
                for eng in df_val_set['ESN'].unique():
                    eng_data = df_val_set[df_val_set['ESN'] == eng]
                    for seg in eng_data['Filter_Segment'].unique():
                        seg_data = eng_data[eng_data['Filter_Segment'] == seg][['P_Scaled', 'Age_Scaled']].values
                        if len(seg_data) > WINDOW_SIZE:
                            seqs = build_sequences_strided(seg_data, WINDOW_SIZE)
                            if len(seqs):
                                X_val_parts.append(seqs)
                                y_val_parts.append(seg_data[WINDOW_SIZE:, 0])

                if X_val_parts:
                    X_val_seq = np.concatenate(X_val_parts, axis=0)
                    y_val_seq = np.concatenate(y_val_parts, axis=0)
                else:
                    X_val_seq = np.array([])
                    y_val_seq = np.array([])

                model          = None
                train_time_min = 0.0
                history_data   = None

                if os.path.exists(MODEL_PATH) and not force_retrain:
                    status_text.text("🧠 Loading Pre-trained AI Model (Fast Mode)...")
                    t0 = time.time()
                    try:
                        model = tf.keras.models.load_model(MODEL_PATH)
                        if os.path.exists(HISTORY_PATH):
                            with open(HISTORY_PATH, 'r') as f: 
                                history_data = json.load(f)
                        train_time_min = (time.time() - t0) / 60.0
                        progress_bar.progress(50)
                    except Exception as e:
                        st.warning("⚠️ File model (.keras) rusak atau tidak kompatibel dengan lingkungan ini. Sistem memaksa Retrain...")
                        force_retrain = True
                        model = None

                if model is None or force_retrain:
                    status_text.text("🧠 Training Neural Network from Scratch...")
                    X_parts, y_parts = [], []
                    for eng in df_train_set['ESN'].unique():
                        eng_data = df_train_set[df_train_set['ESN'] == eng]
                        for seg in eng_data['Filter_Segment'].unique():
                            seg_data = eng_data[eng_data['Filter_Segment'] == seg][['P_Scaled', 'Age_Scaled']].values
                            if len(seg_data) > WINDOW_SIZE:
                                seqs = build_sequences_strided(seg_data, WINDOW_SIZE)
                                if len(seqs):
                                    X_parts.append(seqs)
                                    y_parts.append(seg_data[WINDOW_SIZE:, 0])

                    if X_parts:
                        X_train_seq = np.concatenate(X_parts, axis=0)
                        y_train_seq = np.concatenate(y_parts, axis=0)

                        model = tf.keras.Sequential([
                            tf.keras.layers.GRU(64, return_sequences=True, input_shape=(WINDOW_SIZE, 2)),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.GRU(32, return_sequences=False),
                            tf.keras.layers.Dense(16, activation='relu'),
                            tf.keras.layers.Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')
                        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        t0         = time.time()

                        history = model.fit(
                            X_train_seq, y_train_seq,
                            validation_data=(X_val_seq, y_val_seq) if len(X_val_seq) > 0 else None,
                            epochs=50, batch_size=128, verbose=0,
                            callbacks=[early_stop, StreamlitCallback(progress_bar, status_text, 50)]
                        )
                        train_time_min = (time.time() - t0) / 60.0
                        history_data   = history.history
                        
                        try:
                            model.save(MODEL_PATH)
                            with open(HISTORY_PATH, 'w') as f: json.dump(history_data, f)
                        except Exception: pass

                _gru_step_compiled = None
                if model is not None:
                    _gru_step_compiled = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

                _p_scale   = float(scaler_p.scale_[0])
                _p_min_off = float(scaler_p.min_[0])
                def _scale_p(x: float) -> float: return x * _p_scale + _p_min_off
                def _unscale_p(x_s: float) -> float: return (x_s - _p_min_off) / _p_scale

                status_text.text("🎓 Calculating Global Academic Metrics (Thesis Requirement)...")
                if len(X_val_seq) > 0 and model is not None:
                    try:
                        y_pred_scaled = model.predict(X_val_seq, verbose=0)
                    except ValueError as e:
                        st.error("🚨 **KETIDAKCOCOKAN DIMENSI MODEL (.keras)** 🚨")
                        st.warning("File `arcs_gru_model.keras` yang Anda unggah memiliki jumlah dimensi atau input yang tidak kompatibel dengan kode versi ini (ValueError: assert_input_compat).\n\n**Solusi Wajib:**\nSilakan centang kotak **'Force Retrain AI Model'** di atas, lalu klik **Run Analysis** kembali.")
                        st.stop()

                    y_pred_real   = scaler_p.inverse_transform(y_pred_scaled).flatten()
                    y_val_real    = scaler_p.inverse_transform(y_val_seq.reshape(-1, 1)).flatten()

                    g_rmse    = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
                    g_mae     = mean_absolute_error(y_val_real, y_pred_real)
                    g_mape    = mean_absolute_percentage_error(y_val_real, y_pred_real) * 100
                    g_err_pct = (np.mean(np.abs(y_val_real - y_pred_real)) / np.mean(y_val_real)) * 100

                    thresh       = 10.0
                    y_val_class  = (y_val_real  >= thresh).astype(int)
                    y_pred_class = (y_pred_real >= thresh).astype(int)

                    g_acc  = accuracy_score (y_val_class, y_pred_class)
                    g_prec = precision_score(y_val_class, y_pred_class, zero_division=0)
                    g_rec  = recall_score   (y_val_class, y_pred_class, zero_division=0)
                    g_f1   = f1_score       (y_val_class, y_pred_class, zero_division=0)
                    g_cm   = confusion_matrix(y_val_class, y_pred_class, labels=[0, 1])

                    if len(np.unique(y_val_class)) > 1:
                        fpr, tpr, _ = roc_curve(y_val_class, y_pred_real)
                        g_auc = auc(fpr, tpr)
                    else: fpr, tpr, g_auc = None, None, np.nan

                    feature_names = ['Pressure (PSI)', 'Filter Age']
                    importances   = {}
                    for i in range(2):
                        X_shuff              = X_val_seq.copy()
                        shuffle_idx          = np.random.permutation(len(X_shuff))
                        X_shuff[:, :, i]     = X_val_seq[shuffle_idx, :, i]
                        pred_shuff           = model.predict(X_shuff, verbose=0)
                        pred_shuff_real      = scaler_p.inverse_transform(pred_shuff).flatten()
                        shuff_rmse           = np.sqrt(mean_squared_error(y_val_real, pred_shuff_real))
                        importances[feature_names[i]] = shuff_rmse - g_rmse

                    cv_scores  = []
                    chunk_size = len(X_val_seq) // 4
                    if chunk_size > 0:
                        for i in range(4):
                            s = i * chunk_size
                            e = (i + 1) * chunk_size if i < 3 else len(X_val_seq)
                            cv_scores.append(np.sqrt(mean_squared_error(y_val_real[s:e], y_pred_real[s:e])))

                    st.session_state['thesis_metrics'] = {
                        'rmse': g_rmse, 'mae': g_mae, 'mape': g_mape, 'err_pct': g_err_pct,
                        'acc': g_acc, 'prec': g_prec, 'rec': g_rec, 'f1': g_f1, 'auc': g_auc,
                        'cm': g_cm, 'fpr': fpr, 'tpr': tpr,
                        'y_val_real': y_val_real, 'y_pred_real': y_pred_real,
                        'train_time': train_time_min, 'history': history_data,
                        'importances': importances, 'cv_scores': cv_scores
                    }

                forecast_report = []
                total_engines   = len(engines)

                for idx, eng in enumerate(engines):
                    current_step = idx + 1
                    elapsed_time = time.time() - start_time
                    avg_time     = elapsed_time / current_step
                    eta_seconds  = avg_time * (total_engines - current_step)
                    elapsed_str  = str(timedelta(seconds=int(elapsed_time)))
                    eta_str      = str(timedelta(seconds=int(eta_seconds)))

                    status_text.text(f"⚡ Running PGML Hybrid Forecast... ({current_step}/{total_engines}) | Elapsed: {elapsed_str} | ETA: {eta_str}")
                    progress_bar.progress(50 + int((current_step / total_engines) * 50))

                    df_eng_all       = df_by_esn[eng]
                    df_eng_takeoff   = df_takeoff_by_esn[eng]
                    df_cruise        = df_cruise_by_esn[eng]
                    df_climb         = df_climb_by_esn[eng]
                    eng_processed_data = processed_by_esn[eng]

                    aircraft_reg   = "N/A"
                    possible_regs  = df_eng_all['Aircraft_ID'].dropna() if 'Aircraft_ID' in df_eng_all.columns else pd.Series([], dtype=str)
                    if not possible_regs.empty: aircraft_reg = str(possible_regs.iloc[-1])

                    last_date          = df_eng_takeoff['Date'].iloc[-1] if not df_eng_takeoff.empty else dataset_latest_date
                    start_window_date  = last_date - timedelta(days=30)
                    df_window          = df_eng_takeoff[df_eng_takeoff['Date'] >= start_window_date]
                    recent_data_count  = len(df_window)

                    health_msg        = "NORMAL"
                    insufficient_data = False
                    anomaly_flag      = False

                    if recent_data_count < MIN_DATA_POINTS_30D:
                        health_msg        = "INSUFFICIENT DATA"
                        insufficient_data = True

                    shift_psi = curr_psi = ref_date_val = 0.0
                    ref_date_str = obs_date_str = last_flight_to = last_flight_cr = "-"
                    obs_date_val = None

                    if not df_window.empty:
                        min_idx       = df_window['Pressure_Final'].idxmin()
                        baseline_row  = df_window.loc[min_idx]
                        baseline_psi  = baseline_row['Pressure_Final']
                        ref_date_val  = baseline_row['Date']
                        curr_row     = df_window.iloc[-1]
                        curr_psi     = curr_row['Pressure_Final']
                        obs_date_val = curr_row['Date']
                        shift_psi    = curr_psi - baseline_psi
                        ref_date_str = ref_date_val.strftime('%d-%b-%Y')
                        obs_date_str = obs_date_val.strftime('%d-%b-%Y')

                        if curr_psi > GE_SOFT_LIMIT and shift_psi > RAPID_SHIFT_THRESHOLD: health_msg = "CRITICAL (CNR Triggered)"
                        elif shift_psi > RAPID_SHIFT_THRESHOLD: health_msg = "WARNING (Rapid Shift)"
                    else: curr_psi = df_eng_takeoff['Pressure_Final'].iloc[-1] if not df_eng_takeoff.empty else 0

                    if not df_window.empty and len(df_window) > 10:
                        iso       = IsolationForest(contamination='auto', random_state=42)
                        anomalies = iso.fit_predict(df_window[['Pressure_Final']])
                        scores    = iso.decision_function(df_window[['Pressure_Final']])
                        current_noise = df_window['Pressure_Final'].std()
                        if pd.isna(current_noise): current_noise = 0.0

                        if   curr_psi < 8.0:  noise_tolerance = 0.4
                        elif curr_psi < 11.0: noise_tolerance = 0.8
                        else:                 noise_tolerance = 1.5

                        if (-1 in anomalies) and (min(scores) < -0.05) and (current_noise > noise_tolerance): anomaly_flag = True

                    if anomaly_flag and "CRITICAL" not in health_msg: health_msg = "WARNING (Sensor Anomaly Suspected)"

                    last_to_row = df_eng_all[df_eng_all['Phase'] == 'TAKEOFF']['Date'].max()
                    last_cr_row = df_eng_all[df_eng_all['Phase'] == 'CRUISE' ]['Date'].max()
                    if pd.notna(last_to_row): last_flight_to = last_to_row.strftime('%d-%b-%Y')
                    if pd.notna(last_cr_row): last_flight_cr = last_cr_row.strftime('%d-%b-%Y')

                    days_inactive = (dataset_latest_date - last_date).days
                    is_parked     = days_inactive > 30
                    if is_parked: health_msg = "PARKED"

                    cruise_trending_up = False
                    if not df_cruise.empty:
                        last_cruise = df_cruise['Pressure_Final'].tail(5).mean()
                        prev_cruise = df_cruise['Pressure_Final'].tail(10).head(5).mean()
                        if last_cruise > (prev_cruise + 0.5): cruise_trending_up = True

                    if "WARNING" in health_msg and cruise_trending_up: health_msg = "CRITICAL (Multi-Phase Confirmed)"

                    cnr_table_data = []
                    for ph in ['CRUISE', 'TAKEOFF', 'CLIMB']:
                        df_ph        = _phase_lookup[ph][eng]
                        df_ph_window = df_ph[df_ph['Date'] >= start_window_date]
                        if not df_ph_window.empty and ref_date_val is not None:
                            nearest_idx = (df_ph_window['Date'] - ref_date_val).abs().idxmin()
                            val_ref     = df_ph_window.loc[nearest_idx, 'Pressure_Final']
                            val_obs     = df_ph_window['Pressure_Final'].iloc[-1]
                            change      = val_obs - val_ref
                            cnr_table_data.append({"Flight Phase": ph, "Value at Ref. Date": f"{val_ref:.4f}", "Value at Obs. Date": f"{val_obs:.4f}", "Overall Change": f"{change:.4f}"})
                        else: cnr_table_data.append({"Flight Phase": ph, "Value at Ref. Date": "-", "Value at Obs. Date": "-", "Overall Change": "-"})

                    pred_date_str = planner_date_str = "-"
                    cycles_left = hours_left = 9999
                    final_curve = upper = lower = []
                    engine_rmse = engine_mae = engine_acc = 0.0
                    is_safe_zone = (curr_psi < LOW_PRESSURE_SAFEGUARD)

                    eng_thesis_dict = None

                    if len(eng_processed_data) >= WINDOW_SIZE + 5 and model is not None:
                        try:
                            split_idx_eng  = int(len(eng_processed_data) * 0.8)
                            test_start_idx = max(0, split_idx_eng - WINDOW_SIZE)
                            eval_data      = eng_processed_data.iloc[test_start_idx:].copy()

                            if len(eval_data) > WINDOW_SIZE:
                                eval_arr     = np.hstack([eval_data[['P_Scaled']].values, eval_data[['Age_Scaled']].values])                                           
                                X_eval       = build_sequences_strided(eval_arr, WINDOW_SIZE) 
                                actuals_real = eval_data['Pressure_Final'].values[WINDOW_SIZE:]

                                preds_scaled  = model.predict(X_eval, verbose=0)
                                preds_real    = scaler_p.inverse_transform(preds_scaled).flatten()

                                engine_mae  = np.mean(np.abs(actuals_real - preds_real))
                                engine_rmse = np.sqrt(np.mean(np.square(actuals_real - preds_real)))
                                mean_actual = np.mean(actuals_real)

                                if mean_actual > 0: engine_acc = max(0.0, min(100.0, 100.0 - ((engine_mae / mean_actual) * 100)))

                                e_mape    = mean_absolute_percentage_error(actuals_real, preds_real) * 100 if mean_actual > 0 else 0
                                e_err_pct = (np.mean(np.abs(actuals_real - preds_real)) / mean_actual) * 100 if mean_actual > 0 else 0

                                thresh        = 10.0
                                y_val_class_e  = (actuals_real >= thresh).astype(int)
                                y_pred_class_e = (preds_real   >= thresh).astype(int)

                                e_acc  = accuracy_score (y_val_class_e, y_pred_class_e)
                                e_prec = precision_score(y_val_class_e, y_pred_class_e, zero_division=0)
                                e_rec  = recall_score   (y_val_class_e, y_pred_class_e, zero_division=0)
                                e_f1   = f1_score       (y_val_class_e, y_pred_class_e, zero_division=0)
                                e_cm   = confusion_matrix(y_val_class_e, y_pred_class_e, labels=[0, 1])

                                if len(np.unique(y_val_class_e)) > 1:
                                    e_fpr, e_tpr, _ = roc_curve(y_val_class_e, preds_real)
                                    e_auc = auc(e_fpr, e_tpr)
                                else: e_fpr = e_tpr = None; e_auc = np.nan

                                e_importances = {}
                                for i in range(2):
                                    X_shuff          = X_eval.copy()
                                    shuffle_idx      = np.random.permutation(len(X_shuff))
                                    X_shuff[:, :, i] = X_eval[shuffle_idx, :, i]
                                    pred_shuff       = model.predict(X_shuff, verbose=0)
                                    pred_shuff_real  = scaler_p.inverse_transform(pred_shuff).flatten()
                                    shuff_rmse       = np.sqrt(mean_squared_error(actuals_real, pred_shuff_real))
                                    e_importances[['Pressure (PSI)', 'Filter Age'][i]] = shuff_rmse - engine_rmse

                                e_cv_scores = []
                                chunk_size  = len(X_eval) // 4
                                if chunk_size > 0:
                                    for i in range(4):
                                        s = i * chunk_size
                                        e = (i + 1) * chunk_size if i < 3 else len(X_eval)
                                        e_cv_scores.append(np.sqrt(mean_squared_error(actuals_real[s:e], preds_real[s:e])))

                                eng_thesis_dict = {
                                    'rmse': engine_rmse, 'mae': engine_mae, 'mape': e_mape, 'err_pct': e_err_pct,
                                    'acc': e_acc, 'prec': e_prec, 'rec': e_rec, 'f1': e_f1, 'auc': e_auc, 'cm': e_cm,
                                    'fpr': e_fpr, 'tpr': e_tpr, 'y_val_real': actuals_real, 'y_pred_real': preds_real,
                                    'importances': e_importances, 'cv_scores': e_cv_scores
                                }
                        except Exception: pass

                    if insufficient_data or is_parked or "CRITICAL" in health_msg:
                        final_curve = np.array([curr_psi] * PREDICT_STEPS)
                        upper = lower = final_curve
                        if "CRITICAL" in health_msg:
                            cycles_left = hours_left = 0
                            pred_date_str    = "IMMEDIATE"
                            planner_date_str = "IMMEDIATE (AOG)"
                    else:
                        if len(eng_processed_data) >= WINDOW_SIZE and model is not None and _gru_step_compiled is not None:
                            last_seg_data   = eng_processed_data.tail(WINDOW_SIZE).copy()
                            last_p_scaled   = last_seg_data[['P_Scaled']].values    
                            last_age_scaled = last_seg_data[['Age_Scaled']].values  

                            buf_total = WINDOW_SIZE + PREDICT_STEPS
                            seq_buf   = np.empty((buf_total, 2), dtype=np.float64)
                            seq_buf[:WINDOW_SIZE] = np.hstack([last_p_scaled, last_age_scaled])
                            buf_ptr = WINDOW_SIZE

                            base_curve    = np.empty(PREDICT_STEPS, dtype=np.float64)
                            curr_psi_loop = curr_psi

                            curr_epsilon = INITIAL_POROSITY
                            struct_term  = ((1 - curr_epsilon)**2) / (curr_epsilon**3)
                            k_system     = max(curr_psi_loop, 0.1) / struct_term

                            for i in range(PREDICT_STEPS):
                                curr_seq = seq_buf[buf_ptr - WINDOW_SIZE : buf_ptr].reshape(1, WINDOW_SIZE, 2)
                                
                                try:
                                    next_scaled = float(_gru_step_compiled(curr_seq)[0, 0])
                                except Exception:
                                    st.error("🚨 **KETIDAKCOCOKAN DIMENSI MODEL (.keras)** 🚨")
                                    st.warning("Terjadi error saat Forecasting (ValueError: input_spec). Model Keras lama Anda tidak kompatibel dengan data ini.\n\n**Solusi:** Centang 'Force Retrain AI Model' dan jalankan ulang.")
                                    st.stop()

                                next_p_ai   = _unscale_p(next_scaled)

                                ai_drift = (next_p_ai - curr_psi_loop) * TAKEOFF_DAMPING_FACTOR
                                if ai_drift < MIN_DRIFT_RATE: ai_drift = MIN_DRIFT_RATE

                                blocking_effect = ai_drift * (1 / (curr_epsilon + 0.05))
                                if curr_psi_loop < 9.0: blocking_effect = min(blocking_effect, 0.0001)
                                elif curr_psi_loop < 11.0: blocking_effect = min(blocking_effect, 0.001)

                                curr_epsilon -= blocking_effect
                                if curr_epsilon < 0.05: curr_epsilon = 0.05

                                new_struct_term = ((1 - curr_epsilon)**2) / (curr_epsilon**3)
                                next_p_physics  = k_system * new_struct_term
                                physics_drift   = next_p_physics - curr_psi_loop
                                if physics_drift < MIN_DRIFT_RATE: physics_drift = MIN_DRIFT_RATE

                                trust_physics  = min(1.0, max(0.0, (curr_psi_loop - 9.0) / 4.0))
                                blended_drift  = (1 - trust_physics) * ai_drift + trust_physics * physics_drift
                                next_p_hybrid  = curr_psi_loop + blended_drift

                                base_curve[i]   = next_p_hybrid
                                curr_psi_loop   = next_p_hybrid

                                next_scaled_clamped        = _scale_p(curr_psi_loop)
                                next_age                   = seq_buf[buf_ptr - 1, 1] + 0.005
                                seq_buf[buf_ptr, 0]        = next_scaled_clamped
                                seq_buf[buf_ptr, 1]        = next_age
                                buf_ptr                   += 1

                                if curr_psi_loop >= MAX_Y_LIMIT:
                                    rem = PREDICT_STEPS - i - 1
                                    if rem > 0: base_curve[i + 1:] = curr_psi_loop + np.arange(1, rem + 1) * MIN_DRIFT_RATE
                                    break
                            
                            mc_results  = vectorized_monte_carlo(curr_psi, base_curve, MC_ITERATIONS, PREDICT_STEPS)
                            upper       = np.percentile(mc_results, PARETO_CONFIDENCE,       axis=0)
                            lower       = np.percentile(mc_results, 100 - PARETO_CONFIDENCE, axis=0)
                            final_curve = np.mean(mc_results, axis=0)
                            cross_idx = np.argmax(upper >= CUSTOM_TARGET)

                            if upper[cross_idx] >= CUSTOM_TARGET:
                                cycles_left = int(cross_idx)
                                hours_left  = cycles_left * HOURS_PER_CYCLE
                                days_left   = cycles_left / CYCLES_PER_DAY
                                cross_idx_mean = np.argmax(final_curve >= CUSTOM_TARGET)
                                cycles_late    = int(cross_idx_mean) if final_curve[cross_idx_mean] >= CUSTOM_TARGET else int(cycles_left * 1.1)

                                d_start      = last_date + timedelta(days=int(days_left))
                                d_end        = last_date + timedelta(days=int(cycles_late / CYCLES_PER_DAY))
                                planner_date = d_start - timedelta(days=14)

                                if planner_date <= datetime.now(): planner_date_str = "ACTION REQUIRED NOW"
                                else: planner_date_str = planner_date.strftime('%d %b %Y')
                                pred_date_str = (d_start.strftime('%d %b %Y') if d_start == d_end else f"{d_start.strftime('%d %b')} - {d_end.strftime('%d %b %Y')}")
                            else: pred_date_str = "> 1.5 Years"; planner_date_str = "N/A"
                        else:
                            final_curve = np.array([curr_psi] * PREDICT_STEPS)
                            upper = lower = final_curve

                    if not is_parked and not insufficient_data and "CRITICAL" not in health_msg:
                        if is_safe_zone:
                            if "WARNING" not in health_msg: health_msg = "NORMAL"
                        else:
                            if   hours_left < 100:   health_msg = "CRITICAL"
                            elif hours_left < 300:   health_msg = "WARNING"
                            elif curr_psi >= 13.0:   health_msg = "ON WATCH"

                    degradation_rate = 0.0
                    if ref_date_val and obs_date_val and ref_date_val != obs_date_val:
                        days_diff = (obs_date_val - ref_date_val).days
                        if days_diff > 0: degradation_rate = shift_psi / days_diff

                    forecast_report.append({
                        'ESN': eng, 'Reg': aircraft_reg, 'PSI': round(curr_psi, 2), 'Status': health_msg,
                        'Anomaly': anomaly_flag, 'Date': pred_date_str, 'Planner Date': planner_date_str,
                        'Cycles': cycles_left, 'Hours': hours_left, 'Last Capture': last_date.strftime('%d-%b-%Y'),
                        'Is Parked': is_parked, 'Data Density': recent_data_count, 'CNR Table': cnr_table_data,
                        'Dates Info': {'Ref Date': ref_date_str, 'Obs Date': obs_date_str, 'Last Flight TO': last_flight_to, 'Last Flight CR': last_flight_cr},
                        'Reliability': {'Degradation Rate': degradation_rate, 'Model RMSE': engine_rmse, 'Model MAE': engine_mae, 'Accuracy': engine_acc},
                        'Thesis': eng_thesis_dict, 'Plot Data': (df_eng_takeoff, df_cruise, df_climb, final_curve, upper, lower, last_date), 'Original DF': df_eng_all
                    })

                st.session_state['results']  = forecast_report
                st.session_state['engines']  = engines
                
                try:
                    session_to_save = {
                        "run_time": st.session_state['run_time'],
                        "analyzer_name": st.session_state['analyzer_name'],
                        "engines": engines,
                        "results": forecast_report,
                        "thesis_metrics": st.session_state.get('thesis_metrics')
                    }
                    save_analysis_session(nav_engine, session_to_save)
                except Exception as e:
                    st.warning(f"Failed to auto-save session: {e}")
                
                progress_bar.empty()
                status_text.success(f"✅ Analysis Complete! Total Elapsed Time: {elapsed_str}")
            else: st.error("Please upload the data file first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # 7. TAMPILAN DETIL OPERASIONAL (SINGLE ENGINE)
    # ==========================================
    if st.session_state.get('results') is not None:
        results      = st.session_state['results']
        engines_list = st.session_state.get('engines', [])
        run_time     = st.session_state.get('run_time', '-')
        analyzer_nm  = st.session_state.get('analyzer_name', 'System')
        df_res       = pd.DataFrame(results)

        st.markdown(f"**Session Analysis performed on:** {run_time} (UTC)")
        st.markdown(f"**Analyzed by:** {analyzer_nm}")
        
        # --- FLEET HEALTH OVERVIEW ---
        col1, col2, col3, col4 = st.columns(4)
        crit   = len(df_res[df_res['Status'].str.contains('CRITICAL')])
        warn   = len(df_res[df_res['Status'].str.contains('WARNING')])
        watch  = len(df_res[df_res['Status'].str.contains('ON WATCH')])
        parked = len(df_res[df_res['Status'] == 'PARKED'])

        col1.metric("Total Engines",       len(engines_list))
        col2.metric("Critical",            crit,  delta="-Urgent",  delta_color="inverse")
        col3.metric("Warning / Watch",     warn + watch, delta="Monitor", delta_color="off")
        col4.metric("Healthy / Parked",    len(engines_list) - crit - warn - watch, delta=f"{parked} Parked")

        st.markdown("---")
        st.subheader("📋 Executive Fleet Summary")
        df_res.index = np.arange(1, len(df_res) + 1)
        st.dataframe(df_res[['ESN', 'Reg', 'PSI', 'Status', 'Planner Date', 'Date', 'Cycles']].style.map(color_status, subset=['Status']), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Single Engine Detail (Multi-Phase Analysis)")

        col_sel, col_chart = st.columns([1, 3])

        with col_chart:
            selected_esn_temp = st.session_state.get('sel_esn', engines_list[0])
            res_chart         = next(r for r in results if r['ESN'] == selected_esn_temp)
            df_to, df_cr, df_cl, final_curve, upper, lower, last_date = res_chart['Plot Data']
            future_dates = [last_date + timedelta(days=i / CYCLES_PER_DAY) for i in range(PREDICT_STEPS)]
            x_range_limit = [df_to['Date'].min(), future_dates[-1]] if not df_to.empty else []

            fig1 = go.Figure()
            if not res_chart['Is Parked'] and "INSUFFICIENT" not in res_chart['Status']:
                fig1.add_trace(go.Scatter(x=future_dates, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig1.add_trace(go.Scatter(x=future_dates, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(width=0), name='Confidence', hoverinfo='skip'))
                fig1.add_trace(go.Scatter(x=future_dates, y=final_curve, mode='lines', name='Forecast', line=dict(color='#dc3545', width=2, dash='dash')))
                if res_chart['Cycles'] != 9999:
                    fig1.add_annotation(xref="paper", yref="paper", x=0.02, y=0.9, text=f"<b>Est. Replace:</b> {res_chart['Date']}<br>Earliest Rem. Cycles: {res_chart['Cycles']}", showarrow=False, bgcolor="white", bordercolor="red", font=dict(color="red"))

            fig1.add_trace(go.Scatter(x=df_to['Date'], y=df_to['Pressure_Final'], mode='lines', name='Takeoff History (Raw)', line=dict(color='black', width=1.5)))
            if x_range_limit:
                fig1.add_trace(go.Scatter(x=x_range_limit, y=[14]*2, mode='lines', line=dict(color='rgba(0,123,255,0.4)',   dash='solid'), name='Limit 14 PSI'))
                fig1.add_trace(go.Scatter(x=x_range_limit, y=[18]*2, mode='lines', line=dict(color='rgba(255,193,7,0.4)',   dash='solid'), name='Warning 18 PSI'))
                fig1.add_trace(go.Scatter(x=x_range_limit, y=[27]*2, mode='lines', line=dict(color='rgba(253,126,20,0.4)',  dash='solid'), name='Replace Next Flight 27 PSI'))
                fig1.add_trace(go.Scatter(x=x_range_limit, y=[33]*2, mode='lines', line=dict(color='rgba(0,0,0,0.4)',       dash='solid'), name='Bypass 33 PSI'))
            fig1.update_layout(title="Phase: TAKEOFF (Main Analysis - Raw)", height=350, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", template="plotly_white", yaxis=dict(range=[0, 35], title="PSID"))
            st.plotly_chart(fig1, use_container_width=True)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig2 = go.Figure()
                if not df_cr.empty: fig2.add_trace(go.Scatter(x=df_cr['Date'], y=df_cr['Pressure_Final'], mode='lines', name='Cruise', line=dict(color='#007bff')))
                fig2.update_layout(title="Phase: CRUISE (Correlation)", height=250, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white", yaxis=dict(range=[0, 20]))
                st.plotly_chart(fig2, use_container_width=True)
            with col_c2:
                fig3 = go.Figure()
                if not df_cl.empty: fig3.add_trace(go.Scatter(x=df_cl['Date'], y=df_cl['Pressure_Final'], mode='lines', name='Climb', line=dict(color='#28a745')))
                fig3.update_layout(title="Phase: CLIMB (Correlation)", height=250, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white", yaxis=dict(range=[0, 25]))
                st.plotly_chart(fig3, use_container_width=True)

        with col_sel:
            st.markdown("**Select Engine:**")
            selected_esn = st.selectbox("ESN", engines_list, label_visibility="collapsed", key='sel_esn')
            res = next(r for r in results if r['ESN'] == selected_esn)

            badge_color = "#28a745"
            if   "CRITICAL"    in res['Status']: badge_color = "#dc3545"
            elif "WARNING"     in res['Status']: badge_color = "#ffc107"
            elif "ON WATCH"    in res['Status']: badge_color = "#006400"
            elif "PARKED"      in res['Status']: badge_color = "#6c757d"
            elif "INSUFFICIENT" in res['Status']: badge_color = "#17a2b8"

            inactive_info = ""
            if res['Is Parked']: inactive_info  = "<br><span style='color:red; font-size:11px; font-weight:bold;'>⚠️ Inactive</span>"
            if res['Anomaly']:   inactive_info += "<br><span style='color:#ffc107; font-size:11px; font-weight:bold;'>⚠️ Sensor Anomaly Detected</span>"

            st.markdown(f"""
            <div style="background-color:white;padding:20px;border-radius:10px;border:1px solid #e0e0e0;box-shadow:0 2px 4px rgba(0,0,0,0.05);">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div><h2 style="margin:0;color:#002561;font-size:24px;">{selected_esn}</h2><span style="font-size:14px;color:#666;">({res['Reg']})</span></div>
              </div>
              <div style="margin-bottom:10px;"><span style="background-color:{badge_color};color:white;padding:5px 12px;border-radius:15px;font-size:12px;font-weight:bold;">{res['Status']}</span></div>
              <div style="margin-bottom:15px;">
                <p style="margin:0;color:#666;font-size:12px;text-transform:uppercase;">Current Pressure (T.O)</p>
                <span style="font-size:28px;font-weight:bold;color:#333;">{res['PSI']} <span style="font-size:14px;color:#888;font-weight:normal;">PSID</span></span>
                {inactive_info}
              </div>
              <hr style="margin:10px 0;border:0;border-top:1px solid #eee;">
              <div style="font-size:13px;margin-bottom:10px;background-color:#fff3cd;padding:8px;border-radius:4px;border-left:4px solid #ffc107;">
                <p style="margin:0;color:#856404;font-size:11px;font-weight:600;">PRE-INFO TO PLANNER (14 DAYS PRIOR)</p>
                <p style="margin:0;font-weight:bold;color:#856404;font-size:14px;">{res['Planner Date']}</p>
              </div>
              <div style="font-size:13px;">
                <p style="margin:0;color:#888;font-size:11px;font-weight:600;">REPLACEMENT DUE RANGE</p>
                <p style="margin:0;font-weight:600;color:#005eb8;font-size:16px;">{res['Date']}</p>
                <div style="display:flex;justify-content:space-between;margin-top:5px;"><span>Earliest Rem. Cycles: <b>{res['Cycles']}</b></span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            date_info = res['Dates Info']
            st.markdown(f"""
            <div class='info-box'>
              <b style='color:#002561; font-size:14px;'>Parameter Description (30-Day Shift)</b><hr style="margin-top:5px; margin-bottom:10px;">
              <div class='info-row'><span class='info-label'>Reference Baseline Date:</span><span class='info-val'>{date_info['Ref Date']}</span></div>
              <div class='info-row'><span class='info-label'>Observation Date:</span><span class='info-val'>{date_info['Obs Date']}</span></div>
              <div class='info-row' style='border-bottom:none;'><span class='info-label'>Last Flight Date:</span><span class='info-val'>Cruise: {date_info['Last Flight CR']} <br> Takeoff: {date_info['Last Flight TO']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f8f9fa; padding:25px; border-radius:8px; border:1px solid #e9ecef; margin-bottom:15px;'>", unsafe_allow_html=True)
        st.markdown("<span style='color:#002561; font-weight:bold; font-size:16px; text-transform:uppercase;'>📝 Report Details & Adjustments (PDF)</span><br><br>", unsafe_allow_html=True)
        
        with st.form(key="signature_form"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1: input_name = st.text_input("Name", value=user_display_name)
            with col_s2: input_phone = st.text_input("Phone Number", value="+6281904706205")
            with col_s3: input_email = st.text_input("Email", value="maziz@gmf-aeroasia.co.id")
            
            st.markdown("---")
            uploaded_imgs = st.file_uploader("📎 Upload Supporting Images", type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'], accept_multiple_files=True, help="Upload grafik atau data tambahan sebanyak-banyaknya. Gambar akan dikonversi otomatis agar kompatibel dengan PDF. Format SVG tidak didukung oleh Pillow/FPDF pada deployment ini.")
            
            st.markdown("---")
            use_notes = st.checkbox("➕ Tambahkan Custom Notes Khusus", value=False)
            custom_notes = st.text_area("Tulis catatan Engineer di sini:", disabled=not use_notes)
            
            submit_signature = st.form_submit_button("🔄 Apply All Data to PDF")
            
        st.markdown("</div>", unsafe_allow_html=True)

        img_bytes_list = []
        if uploaded_imgs:
            for img_file in uploaded_imgs:
                if img_file.size > 10 * 1024 * 1024:
                    st.error(f"⚠️ Gambar {img_file.name} melebihi 10 MB. Silakan kompres gambar tersebut.")
                else:
                    img_bytes_list.append(img_file.getvalue())

        notes_text = custom_notes if use_notes else None

        try:
            pdf_bytes = generate_cnr_pdf(
                res, 
                user_name=input_name, 
                user_phone=input_phone, 
                user_email=input_email,
                images_bytes_list=img_bytes_list,
                notes=notes_text
            )
        except Exception as e:
            st.error(f"PDF gagal dibuat: {e}")
            st.stop()

        if not isinstance(pdf_bytes, (bytes, bytearray)) or not bytes(pdf_bytes).startswith(b"%PDF"):
            st.error("PDF gagal dibuat: output bukan file PDF valid.")
            st.stop()

        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            st.download_button(
                label="📄 Download Pre-Info Report (PDF)",
                data=bytes(pdf_bytes),
                file_name=f"PreInfo_{selected_esn}_{datetime.now().strftime('%d%b%Y')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # ==========================================
    # --- 8. TAMPILAN HASIL AKADEMIK (GLOBAL) ---
    # ==========================================
    if st.session_state.get('thesis_metrics') is not None:
        st.markdown("<br><br><hr style='border: 2px solid #005eb8;'>", unsafe_allow_html=True)
        
        with st.expander("🎓 Academic Evaluation (Global Fleet Metrics)", expanded=False):
            st.markdown('<div class="thesis-section">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#002561;margin-top:0;'>Global Fleet Metrics</h2>", unsafe_allow_html=True)
            st.markdown("*Metrik di bawah ini dihitung menggunakan **Validation Set** dari **seluruh engine** yang diuji secara global.*")

            tm = st.session_state['thesis_metrics']
            
            if tm.get('history') is None:
                st.info("⚠️ Riwayat pelatihan (History) tidak ditemukan. Grafik Learning Curve disembunyikan.")
                
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("### 1. Global Regression Metrics")
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("RMSE",  f"{tm['rmse']:.4f}",  delta_color="off")
                rm2.metric("MAE",   f"{tm['mae']:.4f}",   delta_color="off")
                rm3.metric("MAPE",  f"{tm['mape']:.2f}%", delta_color="off")
                st.markdown("<br>", unsafe_allow_html=True)
                rm4, rm5, _ = st.columns(3)
                rm4.metric("% Error (Avg)",  f"{tm['err_pct']:.2f}%",   delta_color="off")
                rm5.metric("Training Time",  f"{tm['train_time']:.2f} min", delta_color="off")

            with col_t2:
                st.markdown("### 2. Global Classification Metrics (Thresh = 10 PSI)")
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("Overall Accuracy", f"{tm['acc']*100:.2f}%")
                cm2.metric("Precision",        f"{tm['prec']*100:.2f}%")
                cm3.metric("Recall",           f"{tm['rec']*100:.2f}%")
                cm4.metric("F1-Score",         f"{tm['f1']*100:.2f}%")
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("AUC (Area Under Curve)", f"{tm['auc']:.4f}")

            st.markdown("---")
            st.markdown("### 3. Global Evaluation Plots")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                df_scatter = pd.DataFrame({'Actual': tm['y_val_real'], 'Predicted': tm['y_pred_real']})
                if len(df_scatter) > 1000: df_scatter = df_scatter.sample(1000, random_state=42)
                fig_scatter = px.scatter(df_scatter, x='Actual', y='Predicted', opacity=0.5, title="Actual vs Predicted (Global)")
                mn, mx = min(df_scatter['Actual'].min(), df_scatter['Predicted'].min()), max(df_scatter['Actual'].max(), df_scatter['Predicted'].max())
                fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
                fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_g")

            with col_p2:
                fig_cm = px.imshow(tm['cm'], text_auto=True, color_continuous_scale='Blues', x=['Normal', 'Degraded'], y=['Normal', 'Degraded'], title="Confusion Matrix (Global)")
                fig_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_cm, use_container_width=True, key="cm_g")

            with col_p3:
                if tm['fpr'] is not None:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=tm['fpr'], y=tm['tpr'], mode='lines', name=f"ROC (AUC={tm['auc']:.2f})", line=dict(color='darkorange', width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='navy', width=2, dash='dash')))
                    fig_roc.update_layout(title="ROC Curve (Global)", margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_roc, use_container_width=True, key="roc_g")
                else: st.info("⚠️ Data validasi tidak mengandung tekanan > 10 PSI. ROC Curve tidak dapat dibuat.")

            st.markdown("<br>", unsafe_allow_html=True)
            col_p4, col_p5, col_p6 = st.columns(3)
            with col_p4:
                if tm.get('history') is not None and 'loss' in tm['history'] and 'val_loss' in tm['history']:
                    epochs = list(range(1, len(tm['history']['loss']) + 1))
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(x=epochs, y=tm['history']['loss'],     mode='lines', name='Train Loss', line=dict(color='blue')))
                    fig_lc.add_trace(go.Scatter(x=epochs, y=tm['history']['val_loss'], mode='lines', name='Val Loss',   line=dict(color='red')))
                    fig_lc.update_layout(title="Learning Curve (Global Model)", margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_lc, use_container_width=True, key="lc_g")

            with col_p5:
                fig_fi = px.bar(x=list(tm['importances'].values()), y=list(tm['importances'].keys()), orientation='h', title="Feature Importance (Global)")
                fig_fi.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_fi, use_container_width=True, key="fi_g")

            with col_p6:
                residuals = tm['y_val_real'] - tm['y_pred_real']
                fig_res   = px.histogram(x=residuals, nbins=50, title="Error Distribution (Global)", color_discrete_sequence=['#28a745'])
                fig_res.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                fig_res.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_res, use_container_width=True, key="res_g")

            if len(tm['cv_scores']) == 4:
                folds  = ['Chron Chunk 1', 'Chron Chunk 2', 'Chron Chunk 3', 'Chron Chunk 4']
                fig_cv = px.bar(x=folds, y=tm['cv_scores'], text=[f"{v:.3f}" for v in tm['cv_scores']], title="Time-Series Validation Stability (Global RMSE)")
                fig_cv.update_traces(textposition='outside', marker_color='#17a2b8')
                fig_cv.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                st.plotly_chart(fig_cv, use_container_width=True, key="cv_g")
            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # --- 9. ENGINE-SPECIFIC ACADEMIC EVALUATION ---
    # ==========================================
    if st.session_state.get('results') is not None:
        results_ref       = st.session_state['results']
        engines_list_ref  = st.session_state.get('engines', [])
        selected_esn_temp = st.session_state.get('sel_esn', engines_list_ref[0] if engines_list_ref else None)
        res_eng = next((r for r in results_ref if r['ESN'] == selected_esn_temp), None)

        if res_eng and res_eng.get('Thesis'):
            tme = res_eng['Thesis']
            
            with st.expander(f"🔍 Engine-Specific Evaluation (ESN: {selected_esn_temp})", expanded=False):
                st.markdown('<div class="engine-thesis-section">', unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#28a745;margin-top:0;'>Engine-Specific Evaluation</h2>", unsafe_allow_html=True)
                st.markdown(f"*Metrik dan plot di bawah ini difilter khusus untuk validasi data dari mesin **{selected_esn_temp}** saja.*")

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("### 1. Specific Regression Metrics")
                    em1, em2, em3 = st.columns(3)
                    em1.metric("RMSE", f"{tme['rmse']:.4f}")
                    em2.metric("MAE",  f"{tme['mae']:.4f}")
                    em3.metric("MAPE", f"{tme['mape']:.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("% Error (Avg)", f"{tme['err_pct']:.2f}%")

                with col_e2:
                    st.markdown("### 2. Specific Classification Metrics")
                    em4, em5, em6, em7 = st.columns(4)
                    em4.metric("Accuracy",  f"{tme['acc']*100:.2f}%")
                    em5.metric("Precision", f"{tme['prec']*100:.2f}%")
                    em6.metric("Recall",    f"{tme['rec']*100:.2f}%")
                    em7.metric("F1-Score",  f"{tme['f1']*100:.2f}%")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("AUC", f"{tme['auc']:.4f}" if not pd.isna(tme['auc']) else "N/A (No >10 PSI data)")

                st.markdown("---")
                st.markdown("### 3. Specific Evaluation Plots")
                col_ep1, col_ep2, col_ep3 = st.columns(3)

                with col_ep1:
                    df_e_sc = pd.DataFrame({'Actual': tme['y_val_real'], 'Predicted': tme['y_pred_real']})
                    if len(df_e_sc) > 1000: df_e_sc = df_e_sc.sample(1000, random_state=42)
                    fig_e_sc = px.scatter(df_e_sc, x='Actual', y='Predicted', opacity=0.6, title=f"Actual vs Predicted ({selected_esn_temp})")
                    mn_e = min(df_e_sc['Actual'].min(), df_e_sc['Predicted'].min())
                    mx_e = max(df_e_sc['Actual'].max(), df_e_sc['Predicted'].max())
                    fig_e_sc.add_trace(go.Scatter(x=[mn_e, mx_e], y=[mn_e, mx_e], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
                    fig_e_sc.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_e_sc, use_container_width=True, key=f"scatter_e_{selected_esn_temp}")

                with col_ep2:
                    fig_e_cm = px.imshow(tme['cm'], text_auto=True, color_continuous_scale='Greens', x=['Normal', 'Degraded'], y=['Normal', 'Degraded'], title=f"Confusion Matrix ({selected_esn_temp})")
                    fig_e_cm.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_e_cm, use_container_width=True, key=f"cm_e_{selected_esn_temp}")

                with col_ep3:
                    if tme['fpr'] is not None:
                        fig_e_roc = go.Figure()
                        fig_e_roc.add_trace(go.Scatter(x=tme['fpr'], y=tme['tpr'], mode='lines', name=f"AUC = {tme['auc']:.2f}", line=dict(color='green', width=2)))
                        fig_e_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', width=2, dash='dash')))
                        fig_e_roc.update_layout(title=f"ROC Curve ({selected_esn_temp})", margin=dict(l=20, r=20, t=40, b=20), height=300)
                        st.plotly_chart(fig_e_roc, use_container_width=True, key=f"roc_e_{selected_esn_temp}")
                    else: st.info("⚠️ Data validasi mesin ini belum pernah melampaui 10 PSI. ROC Curve tidak dapat dihitung.")

                st.markdown("<br>", unsafe_allow_html=True)
                col_ep4, col_ep5, col_ep6 = st.columns(3)
                with col_ep4: st.info("ℹ️ **Model Learning Curve** tidak ditampilkan secara individual karena model AI dilatih secara terpusat (Global Fleet).")

                with col_ep5:
                    fig_e_fi = px.bar(x=list(tme['importances'].values()), y=list(tme['importances'].keys()), orientation='h', title=f"Feature Importance ({selected_esn_temp})", color_discrete_sequence=['#28a745'])
                    fig_e_fi.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_e_fi, use_container_width=True, key=f"fi_e_{selected_esn_temp}")

                with col_ep6:
                    e_res   = tme['y_val_real'] - tme['y_pred_real']
                    fig_e_r = px.histogram(x=e_res, nbins=30, title=f"Error Distribution ({selected_esn_temp})", color_discrete_sequence=['#17a2b8'])
                    fig_e_r.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                    fig_e_r.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_e_r, use_container_width=True, key=f"res_e_{selected_esn_temp}")

                if len(tme['cv_scores']) == 4:
                    fig_e_cv = px.bar(x=['Chunk 1', 'Chunk 2', 'Chunk 3', 'Chunk 4'], y=tme['cv_scores'], text=[f"{v:.3f}" for v in tme['cv_scores']], title=f"Validation Stability over Time ({selected_esn_temp})")
                    fig_e_cv.update_traces(textposition='outside', marker_color='#6f42c1')
                    fig_e_cv.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_e_cv, use_container_width=True, key=f"cv_e_{selected_esn_temp}")
                st.markdown('</div>', unsafe_allow_html=True)
        else: 
            st.warning("⚠️ Data akademis untuk mesin ini tidak cukup untuk di-generate (Minimal butuh 65 data points penerbangan pada porsi Validation Set).")
