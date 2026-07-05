# 📊 Campus Event Attendance Forecaster

A desktop application that predicts how many students will attend an upcoming
campus event — so organizers can book the **right-sized venue** instead of
guessing. Built with **Python, Tkinter (GUI)** and **scikit-learn (Machine
Learning)**.

---

## 🎯 What This Project Does

Every semester, student clubs and event committees have to decide: *"Should
we book the 200-seat seminar hall or the 1000-seat auditorium?"* Getting it
wrong means either an embarrassingly empty hall or an overcrowded,
uncomfortable event.

This app solves that problem. The organizer enters details about the planned
event — its **type, day, time slot, expected weather, marketing effort,
expected social media reach, venue capacity, and past average attendance for
similar events** — and a trained **machine learning model instantly predicts
the expected number of attendees**. The app then:

- Shows the predicted headcount and what % of the venue it would fill
- Gives a clear **venue-booking recommendation** (too small / good fit / too big)
- Draws a bar chart comparing **Past Average vs. Predicted vs. Venue Capacity**

It runs as a real, standalone desktop program with a graphical interface —
not just a script — so it can be demoed and used like actual software.

---

## 🧠 How AI Is Used Here (the important part!)

This project uses **supervised machine learning regression**, implemented
with **scikit-learn**, to turn event details into a numeric attendance
prediction. Concretely:

1. **Training data (`data/generate_data.py`)**
   A dataset of past campus events is used, where each row has event
   features (type, day, time, weather, marketing effort, social reach,
   venue capacity, holiday flag, competing-event flag, past average
   attendance) and the **actual attendance** that resulted. In this project
   the historical dataset is synthetically generated with realistic
   real-world relationships baked in (e.g., Fests draw bigger crowds than
   Seminars, rain reduces turnout, better marketing increases turnout) —
   in a real deployment, you would swap this file's output for your
   college's actual past event records (from registration forms, gate
   counts, etc.), and the rest of the pipeline works unchanged.

2. **Feature engineering**
   Categorical fields (event type, day of week, time slot, weather,
   marketing effort) are converted into numeric form using **one-hot
   encoding** (`pandas.get_dummies`), since ML models need numbers, not
   text labels.

3. **Model: Random Forest Regressor**
   A `RandomForestRegressor` (an ensemble of decision trees, from
   `sklearn.ensemble`) is trained on 80% of the historical data
   (`sklearn.model_selection.train_test_split`). A Random Forest was chosen
   because it:
   - Handles a mix of numeric and one-hot categorical features well
   - Captures non-linear interactions (e.g., "Fest" + "High marketing" +
     "Evening" combining to boost attendance more than any single factor
     alone)
   - Is robust to noise and doesn't require feature scaling

4. **Evaluation**
   The remaining 20% of data (unseen by the model) is used to measure
   accuracy with **Mean Absolute Error (MAE)** and **R² score**
   (`sklearn.metrics`), which are printed to the console during training so
   you can quote real accuracy numbers in your report/LinkedIn post.

5. **Serving predictions in real time**
   The trained model is saved to disk with **`joblib`**. The Tkinter GUI
   loads this saved model once at startup, and every time the user clicks
   **"Predict Attendance"**, their form inputs are converted into the exact
   same encoded feature format the model was trained on, and
   `model.predict(...)` returns an attendance estimate in milliseconds.

In short: **scikit-learn is the brain (the trained regression model),
Tkinter is the face (the interactive GUI)** that lets a non-technical event
organizer use that brain without touching any code.

---

## 🗂️ Project Structure

```
campus_event_attendance_forecaster/
├── app.py                     # Tkinter GUI application (run this!)
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
├── data/
│   ├── generate_data.py         # Generates the historical training dataset
│   └── campus_events.csv        # Historical event data (auto-created)
└── model/
    ├── train_model.py            # Trains & evaluates the scikit-learn model
    ├── attendance_model.pkl       # Saved trained model (auto-created)
    └── feature_columns.pkl        # Saved feature layout (auto-created)
```

---

## 🪟 Windows Quick Start (easiest option)

Two helper scripts are included so you don't need to type any commands:

1. **Double-click `install.bat`** once (installs all required libraries). Wait for it to finish, then press any key.
2. **Double-click `run.bat`** every time you want to launch the app. A window will open and stay open, so if anything goes wrong you'll be able to read the error message instead of it flashing shut.

If you prefer the command line, or are on Mac/Linux, follow the manual steps below instead.

## ⚙️ Installation

**Requirements:** Python 3.9+ (Tkinter ships with most standard Python
installs; on some Linux distros install it separately, e.g.
`sudo apt install python3-tk`).

1. Download / clone this project folder.
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

**Just run the app** — it automatically generates the training data and
trains the model the very first time it's launched (this takes a few
seconds), then reuses the saved model on every future launch:

```bash
python app.py
```

If you'd like to (re)train the model manually, or inspect accuracy metrics
in the console, you can run:

```bash
python model/train_model.py
```

To regenerate a fresh synthetic dataset:

```bash
python data/generate_data.py
```

### Using the App
1. Select the **event type**, **day**, **time slot**, expected **weather**,
   and your planned **marketing effort** from the dropdowns.
2. Enter your **expected social media reach**, the **venue capacity** you're
   considering, and the **past average attendance** for similar events.
3. Tick the checkboxes if the event **falls on a holiday** or if there's a
   **competing event** nearby.
4. Click **🔮 Predict Attendance**.
5. Read the predicted headcount, the % of venue capacity it represents, the
   color-coded **recommendation**, and the comparison chart.

---

## 🛠️ Tech Stack

| Layer               | Technology                                  |
|---------------------|----------------------------------------------|
| GUI                 | Tkinter, ttk                                 |
| Machine Learning    | scikit-learn (`RandomForestRegressor`)       |
| Data handling       | pandas, NumPy                                |
| Model persistence   | joblib                                       |
| Charts              | matplotlib (embedded in the Tkinter window)  |

---

## 🚀 Possible Future Improvements

- Replace the synthetic dataset with real historical attendance records
  exported from your college's event registration system
- Add more features: event duration, ticket price, guest speaker fame,
  department, exam-week proximity
- Try additional models (Gradient Boosting, XGBoost) and compare accuracy
- Add a "batch prediction" mode: upload a CSV of multiple planned events and
  get predictions for all of them at once
- Export the prediction + chart as a PDF report for submission to the venue
  booking office

---

## 📌 About This Project

Built as part of an internship assignment to demonstrate practical use of
machine learning (scikit-learn) combined with desktop GUI development
(Tkinter) to solve a real, everyday campus problem: **smarter venue booking
through data-driven attendance forecasting.**
