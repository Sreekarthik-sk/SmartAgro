import os
import sqlite3
import numpy as np
from flask import Flask, render_template, request, url_for, session, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from passlib.hash import bcrypt   # for password hashing

app = Flask(__name__)
app.secret_key = "smartagro_secret"  # Needed for session storage

# ===== CONFIG =====
MODEL_PATH = "sugarcane_model_improved.keras"
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DB_PATH = "users.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== LOAD MODEL =====
model = load_model(MODEL_PATH)
class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

# ===== DATABASE =====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# ===== UTILS =====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def require_login():
    if "user" not in session:
        return redirect(url_for("login"))
    return None

# ===== ROUTES =====
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if not username or not password:
            return render_template("signup.html", error="Username and password required")

        hashed_pw = bcrypt.hash(password)  # hash before saving

        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("signup.html", error="Username already exists")

    return render_template("signup.html")

# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user and bcrypt.verify(password, user["password"]):
            session["user"] = username
            session.setdefault("history", [])
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("history", None)
    return redirect(url_for("login"))

# ---------- DIAGNOSE ----------
@app.route("/diagnose", methods=["GET", "POST"])
def diagnose():
    gate = require_login()
    if gate:
        return gate

    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file and file.filename != "" and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            img = prepare_image(file_path)
            try:
                preds = model.predict(img)
                prediction = class_names[np.argmax(preds)]
                confidence = round(float(np.max(preds)) * 100, 2)
                image_url = url_for('static', filename=f'uploads/{filename}')

                # Save to session history
                history = session.get("history", [])
                history.append({
                    "filename": filename,
                    "prediction": prediction,
                    "confidence": confidence,
                    "image_url": image_url
                })
                session["history"] = history

            except Exception as e:
                prediction = "Error"
                confidence = 0
                print("Prediction error:", e)

    return render_template(
        "diagnose.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url
    )

# ---------- INFO ----------
@app.route("/info/<disease>")
def info(disease):
    gate = require_login()
    if gate:
        return gate
    return render_template("info.html", disease=disease)

# ---------- ABOUT ----------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------- HISTORY ----------
@app.route("/history")
def history():
    gate = require_login()
    if gate:
        return gate
    return render_template("history.html", history=session.get("history", []))

@app.route("/clear_history")
def clear_history():
    gate = require_login()
    if gate:
        return gate
    session.pop("history", None)
    return redirect(url_for("history"))

# ---------- HEALTH CHECK ----------
@app.route("/ping")
def ping():
    return "pong"

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)