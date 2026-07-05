"""
app.py
------
Campus Event Attendance Forecaster - GUI Application

A desktop application (built with Tkinter) that uses a trained scikit-learn
Random Forest model to predict how many students will attend an upcoming
campus event, based on details the organizer enters (event type, day, time,
weather forecast, marketing effort, expected social reach, venue capacity,
etc). It then gives a plain-English venue-booking recommendation and a small
chart comparing the prediction against venue capacity and past performance.

Run:
    python app.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import joblib

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "attendance_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

EVENT_TYPES = ["Technical", "Cultural", "Sports", "Workshop", "Seminar", "Fest"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_SLOTS = ["Morning", "Afternoon", "Evening"]
WEATHER = ["Sunny", "Cloudy", "Rainy"]
MARKETING = ["Low", "Medium", "High"]

CATEGORICAL_COLS = ["event_type", "day_of_week", "time_slot", "weather", "marketing_effort"]

# ---------- Theme ----------
BG = "#0f1729"
PANEL = "#182238"
ACCENT = "#4f8dfd"
ACCENT_DARK = "#2f5fc4"
TEXT_LIGHT = "#e7ecf7"
TEXT_MUTED = "#93a0b8"
GOOD = "#33c07f"
WARN = "#e0a83a"
BAD = "#e0554a"


def ensure_model_ready():
    """Train the model automatically on first run if it doesn't exist yet."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
        sys.path.insert(0, MODEL_DIR)
        import train_model
        train_model.main()


class AttendanceForecasterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Campus Event Attendance Forecaster")
        self.geometry("980x680")
        self.minsize(900, 640)
        self.configure(bg=BG)

        ensure_model_ready()
        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(COLUMNS_PATH)

        self._build_style()
        self._build_layout()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=BG, foreground=TEXT_LIGHT, font=("Segoe UI", 20, "bold"))
        style.configure("SubHeader.TLabel", background=BG, foreground=TEXT_MUTED, font=("Segoe UI", 10))
        style.configure("ResultTitle.TLabel", background=PANEL, foreground=TEXT_MUTED, font=("Segoe UI", 10, "bold"))
        style.configure("ResultValue.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI", 34, "bold"))

        style.configure("TCombobox", fieldbackground="#0d1526", background="#0d1526",
                         foreground=TEXT_LIGHT, arrowcolor=TEXT_LIGHT)
        style.map("TCombobox", fieldbackground=[("readonly", "#0d1526")])

        style.configure("TEntry", fieldbackground="#0d1526", foreground=TEXT_LIGHT)

        style.configure("Accent.TButton", background=ACCENT, foreground="#0b1220",
                         font=("Segoe UI", 11, "bold"), padding=10, borderwidth=0)
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])

        style.configure("TCheckbutton", background=PANEL, foreground=TEXT_LIGHT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)])

    def _build_layout(self):
        # ---- Header ----
        header = ttk.Frame(self, style="TFrame", padding=(24, 18, 24, 8))
        header.pack(fill="x")
        ttk.Label(header, text="📊 Campus Event Attendance Forecaster", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="AI-powered attendance prediction to help you choose the right venue, every time.",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self, style="TFrame", padding=(24, 8, 24, 24))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        # ---- Left panel: inputs ----
        form_panel = tk.Frame(body, bg=PANEL, bd=0)
        form_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_form(form_panel)

        # ---- Right panel: results + chart ----
        right = ttk.Frame(body, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        result_panel = tk.Frame(right, bg=PANEL, bd=0)
        result_panel.grid(row=0, column=0, sticky="new", pady=(0, 12))
        self._build_result_panel(result_panel)

        chart_panel = tk.Frame(right, bg=PANEL, bd=0)
        chart_panel.grid(row=1, column=0, sticky="nsew")
        self._build_chart_panel(chart_panel)

        # Footer
        footer = ttk.Label(
            self, text="Model: Random Forest Regressor (scikit-learn)  |  GUI: Tkinter",
            style="SubHeader.TLabel", padding=(24, 0, 0, 10)
        )
        footer.pack(anchor="w")

    def _build_form(self, parent):
        # IMPORTANT LAYOUT ORDER: the Predict button is packed to the BOTTOM
        # of `parent` FIRST, before anything else. In Tkinter's pack manager,
        # whatever is packed first with side="bottom" claims its space
        # permanently at the bottom -- guaranteeing the button is always
        # visible, no matter the window size or how many fields there are.
        # Everything else (the scrollable field list) is packed afterward
        # and simply fills whatever space remains above it.

        button_frame = tk.Frame(parent, bg=PANEL)
        button_frame.pack(side="bottom", fill="x", padx=20, pady=(10, 20))
        self.predict_button = ttk.Button(
            button_frame, text="🔮  Predict Attendance", style="Accent.TButton",
            command=self.predict
        )
        self.predict_button.pack(fill="x")

        title = tk.Label(parent, text="Event Details", bg=PANEL, fg=TEXT_LIGHT,
                          font=("Segoe UI", 13, "bold"))
        title.pack(side="top", anchor="w", padx=20, pady=(18, 6))

        # Scrollable area for all the input fields (fills remaining space
        # between the title and the pinned button above).
        scroll_area = tk.Frame(parent, bg=PANEL)
        scroll_area.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(scroll_area, bg=PANEL, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(scroll_area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0))
        scrollbar.pack(side="right", fill="y")

        fields_frame = tk.Frame(canvas, bg=PANEL)
        fields_window = canvas.create_window((0, 0), window=fields_frame, anchor="nw")

        def on_fields_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(fields_window, width=event.width)

        fields_frame.bind("<Configure>", on_fields_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        self.vars = {}
        pad = {"padx": 20, "pady": (10, 4)}

        def add_combo(label, key, values, default=None):
            ttk.Label(fields_frame, text=label, style="Panel.TLabel").pack(anchor="w", **pad)
            var = tk.StringVar(value=default or values[0])
            combo = ttk.Combobox(fields_frame, textvariable=var, values=values, state="readonly")
            combo.pack(fill="x", padx=20)
            self.vars[key] = var

        def add_entry(label, key, default):
            ttk.Label(fields_frame, text=label, style="Panel.TLabel").pack(anchor="w", **pad)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(fields_frame, textvariable=var)
            entry.pack(fill="x", padx=20)
            self.vars[key] = var

        add_combo("Event Type", "event_type", EVENT_TYPES)
        add_combo("Day of Week", "day_of_week", DAYS, default="Saturday")
        add_combo("Time Slot", "time_slot", TIME_SLOTS, default="Evening")
        add_combo("Expected Weather", "weather", WEATHER)
        add_combo("Marketing Effort", "marketing_effort", MARKETING, default="Medium")

        add_entry("Expected Social Media Reach (views)", "social_media_reach", 3000)
        add_entry("Venue Capacity (seats)", "venue_capacity", 500)
        add_entry("Past Average Attendance (similar events)", "past_avg_attendance", 250)

        check_frame = tk.Frame(fields_frame, bg=PANEL)
        check_frame.pack(fill="x", padx=18, pady=(10, 14))
        self.holiday_var = tk.BooleanVar(value=False)
        self.competing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(check_frame, text="Falls on a holiday", variable=self.holiday_var).pack(
            side="left", padx=6)
        ttk.Checkbutton(check_frame, text="Competing event nearby", variable=self.competing_var).pack(
            side="left", padx=6)

    def _build_result_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        pad = dict(padx=20, pady=(16, 4))

        tk.Label(parent, text="PREDICTED ATTENDANCE", bg=PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        self.result_value_label = tk.Label(parent, text="—", bg=PANEL, fg=ACCENT,
                                            font=("Segoe UI", 40, "bold"))
        self.result_value_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=20)

        self.result_sub_label = tk.Label(parent, text="Fill in event details and click Predict",
                                          bg=PANEL, fg=TEXT_MUTED, font=("Segoe UI", 10))
        self.result_sub_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 10))

        self.recommendation_label = tk.Label(
            parent, text="", bg=PANEL, fg=TEXT_LIGHT, font=("Segoe UI", 11, "bold"),
            wraplength=460, justify="left"
        )
        self.recommendation_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 18))

    def _build_chart_panel(self, parent):
        self.figure = Figure(figsize=(5, 3.4), dpi=100, facecolor=PANEL)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=16)
        self._draw_chart(predicted=0, capacity=500, past_avg=0)

    def _style_axes(self):
        self.ax.set_facecolor(PANEL)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_MUTED)
        self.ax.tick_params(colors=TEXT_MUTED)
        self.ax.title.set_color(TEXT_LIGHT)

    # ---------------------------------------------------------------
    # Prediction logic
    # ---------------------------------------------------------------
    def _validate_inputs(self):
        try:
            reach = float(self.vars["social_media_reach"].get())
            capacity = float(self.vars["venue_capacity"].get())
            past_avg = float(self.vars["past_avg_attendance"].get())
            if capacity <= 0:
                raise ValueError("Venue capacity must be greater than 0.")
            if reach < 0 or past_avg < 0:
                raise ValueError("Reach and past attendance cannot be negative.")
            return reach, capacity, past_avg
        except ValueError as e:
            messagebox.showerror("Invalid input", f"Please check your numeric fields.\n\n{e}")
            return None

    def predict(self):
        validated = self._validate_inputs()
        if validated is None:
            return
        reach, capacity, past_avg = validated

        row = {
            "event_type": self.vars["event_type"].get(),
            "day_of_week": self.vars["day_of_week"].get(),
            "time_slot": self.vars["time_slot"].get(),
            "weather": self.vars["weather"].get(),
            "marketing_effort": self.vars["marketing_effort"].get(),
            "social_media_reach": reach,
            "venue_capacity": capacity,
            "is_holiday": int(self.holiday_var.get()),
            "competing_event": int(self.competing_var.get()),
            "past_avg_attendance": past_avg,
        }
        df_row = pd.DataFrame([row])

        encoded = pd.get_dummies(df_row[CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
        numeric_cols = ["social_media_reach", "venue_capacity", "is_holiday",
                         "competing_event", "past_avg_attendance"]
        features = pd.concat([encoded, df_row[numeric_cols]], axis=1)
        features = features.reindex(columns=self.feature_columns, fill_value=0)

        prediction = self.model.predict(features)[0]
        prediction = max(0, min(prediction, capacity))
        fill_pct = (prediction / capacity) * 100

        self.result_value_label.config(text=f"{int(round(prediction))} students")
        self.result_sub_label.config(
            text=f"≈ {fill_pct:.1f}% of venue capacity ({int(capacity)} seats)"
        )

        self._set_recommendation(fill_pct)
        self._draw_chart(predicted=prediction, capacity=capacity, past_avg=past_avg)

    def _set_recommendation(self, fill_pct):
        if fill_pct >= 95:
            text = ("⚠️  Predicted turnout meets or exceeds venue capacity. "
                    "Consider booking a larger venue or arranging an overflow area / live stream.")
            color = BAD
        elif fill_pct >= 75:
            text = ("✅  Great fit! Predicted turnout comfortably fills the venue without overcrowding. "
                    "This venue size looks well matched to demand.")
            color = GOOD
        elif fill_pct >= 40:
            text = ("ℹ️  Predicted turnout is moderate. This venue works, but you could also boost "
                    "marketing effort to increase attendance.")
            color = WARN
        else:
            text = ("⚠️  Predicted turnout is low relative to venue size. Consider a smaller venue "
                    "to keep the event feeling lively, or increase promotional efforts.")
            color = WARN

        self.recommendation_label.config(text=text, fg=color)

    def _draw_chart(self, predicted, capacity, past_avg):
        self.ax.clear()
        self._style_axes()

        labels = ["Past Avg.", "Predicted", "Venue Capacity"]
        values = [past_avg, predicted, capacity]
        colors = [TEXT_MUTED, ACCENT, WARN]

        bars = self.ax.bar(labels, values, color=colors, width=0.55)
        self.ax.set_title("Predicted Attendance vs. Capacity", fontsize=11, pad=10)
        self.ax.set_ylabel("Attendees", color=TEXT_MUTED, fontsize=9)

        for bar, val in zip(bars, values):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                         f"{int(val)}", ha="center", va="bottom", color=TEXT_LIGHT, fontsize=9)

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = AttendanceForecasterApp()
    app.mainloop()
