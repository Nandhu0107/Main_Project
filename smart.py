import tkinter as tk
import random
import time
import math
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

try:
    from main_system import ontology_reasoning
except Exception:
    def ontology_reasoning(congestion_label, emergency_flag):
        if congestion_label == "High" and emergency_flag == 1:
            return {
                "action": "Activate Green Wave",
                "priority": "Critical",
                "reason": "High congestion with emergency vehicle detected"
            }
        elif congestion_label == "High":
            return {
                "action": "Traffic Diversion Recommended",
                "priority": "High",
                "reason": "High congestion level"
            }
        elif congestion_label == "Medium":
            return {
                "action": "Adjust Signal Timing",
                "priority": "Moderate",
                "reason": "Moderate congestion detected"
            }
        return {
            "action": "Normal Monitoring",
            "priority": "Low",
            "reason": "Traffic operating normally"
        }


# ==========================
# UI CONSTANTS
# ==========================
WIDTH = 1000
HEIGHT = 800

APP_BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
BORDER = "#D7DDE5"
TEXT = "#111827"
MUTED = "#6B7280"
ACCENT = "#2563EB"
ACCENT_DARK = "#1E40AF"
ACCENT_SOFT = "#E0E7FF"
CANVAS_BG = "#F8FAFC"
ROAD = "#7B808A"
LINE = "#F5F5F5"
LANE = "#F4C430"

FONT_TITLE = ("Segoe UI Semibold", 18)
FONT_SUB = ("Segoe UI", 10)
FONT_SECTION = ("Segoe UI Semibold", 12)
FONT_BODY = ("Segoe UI", 10)

root = tk.Tk()
root.title("AI Smart City Traffic Control System")
root.configure(bg=APP_BG)
root.minsize(1280, 820)

# ==========================
# HEADER
# ==========================
header = tk.Frame(root, bg=APP_BG)
header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

title_block = tk.Frame(header, bg=APP_BG)
title_block.pack(side=tk.LEFT, anchor="w")
tk.Label(title_block, text="Smart City Traffic Control", bg=APP_BG, fg=TEXT, font=FONT_TITLE).pack(anchor="w")
tk.Label(title_block, text="Ontology-driven Surveillance Console", bg=APP_BG, fg=MUTED, font=FONT_SUB).pack(anchor="w")

badge = tk.Label(header, text="LIVE", bg=ACCENT_SOFT, fg=ACCENT_DARK, font=FONT_SUB, padx=10, pady=4)
badge.pack(side=tk.RIGHT, anchor="e")

# ==========================
# MAIN CONTENT
# ==========================
content = tk.Frame(root, bg=APP_BG)
content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
content.grid_columnconfigure(0, weight=1)
content.grid_columnconfigure(1, weight=0)
content.grid_rowconfigure(0, weight=1)

left_frame = tk.Frame(content, bg=APP_BG)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

right_panel = tk.Frame(content, bg=APP_BG, width=320)
right_panel.grid(row=0, column=1, sticky="ns")
right_panel.grid_propagate(False)

canvas_frame = tk.Frame(left_frame, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
canvas_frame.pack(fill=tk.BOTH, expand=True)
canvas = tk.Canvas(canvas_frame, width=WIDTH, height=HEIGHT, bg=CANVAS_BG, highlightthickness=0)
canvas.pack(padx=12, pady=12)

# ==========================
# DRAW ROADS
# ==========================
canvas.create_rectangle(350, 0, 650, HEIGHT, fill=ROAD)
canvas.create_rectangle(0, 250, WIDTH, 550, fill=ROAD)

canvas.create_line(500, 0, 500, HEIGHT, fill=LANE, width=2, dash=(12, 8))
canvas.create_line(0, 400, WIDTH, 400, fill=LANE, width=2, dash=(12, 8))

STOP_LINES = {"N": 240, "S": 560, "E": 660, "W": 340}

canvas.create_line(350,240,650,240,fill=LINE,width=4)
canvas.create_line(350,560,650,560,fill=LINE,width=4)
canvas.create_line(340,250,340,550,fill=LINE,width=4)
canvas.create_line(660,250,660,550,fill=LINE,width=4)

# ==========================
# TRAFFIC SIGNAL CLASS
# ==========================
class Signal:
    def __init__(self, x, y):
        self.red = canvas.create_oval(x,y,x+20,y+20,fill="black")
        self.yellow = canvas.create_oval(x,y+25,x+20,y+45,fill="black")
        self.green = canvas.create_oval(x,y+50,x+20,y+70,fill="black")
        self.timer_text = canvas.create_text(x+10,y+85,text="",font=FONT_BODY)
        self.state = "RED"
        self.time_left = 0

    def set(self,state,time_left=0):
        self.state = state
        self.time_left = time_left
        canvas.itemconfig(self.red,fill="black")
        canvas.itemconfig(self.yellow,fill="black")
        canvas.itemconfig(self.green,fill="black")
        if state=="RED":
            canvas.itemconfig(self.red,fill="red")
        elif state=="YELLOW":
            canvas.itemconfig(self.yellow,fill="yellow")
        elif state=="GREEN":
            canvas.itemconfig(self.green,fill="green")

    def update_timer(self):
        canvas.itemconfig(self.timer_text,text=str(self.time_left))


signals = {
    "N": Signal(480,200),
    "S": Signal(520,580),
    "E": Signal(680,380),
    "W": Signal(300,380)
}

# ==========================
# (Everything else in your original code remains EXACTLY THE SAME)
# ==========================

# IMPORTANT CHANGE INSIDE log_surveillance()

def log_surveillance():
    global LAST_LOG_TIME, LOG_PATH, LOG_DISABLED
    now = time.time()
    if now - LAST_LOG_TIME < LOG_INTERVAL:
        return
    LAST_LOG_TIME = now
    if LOG_DISABLED:
        return

    write_header = not os.path.exists(LOG_PATH)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:   # <- fixed here
            if write_header:
                f.write("timestamp,...\n")
            f.write("data...\n")
    except Exception:
        LOG_DISABLED = True

# ==========================
# START SYSTEM
# ==========================
signal_cycle()
update()
root.mainloop()
