import os
import sqlite3
from functools import wraps

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Cambia esta clave antes de publicar la aplicación.
app.secret_key = os.environ.get("SECRET_KEY", "clave-solo-para-desarrollo")

DATABASE = os.path.join(app.root_path, "tickets.db")


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'Media',
                status TEXT NOT NULL DEFAULT 'Abierto',
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        """)

        # Cuenta de demostración; se crea solo si no existe.
        user = db.execute(
            "SELECT id FROM users WHERE email = ?",
            ("admin@tickets.com",)
        ).fetchone()

        if user is None:
            db.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (
                    "admin@tickets.com",
                    generate_password_hash("Admin123!")
                )
            )

        # Tickets de ejemplo para mostrar el dashboard.
        count = db.execute(
            "SELECT COUNT(*) AS total FROM tickets"
        ).fetchone()["total"]

        if count == 0:
            db.executemany("""
                INSERT INTO tickets (title, description, priority, status)
                VALUES (?, ?, ?, ?)
            """, [
                (
                    "Error al iniciar sesión",
                    "El usuario no puede acceder a la plataforma.",
                    "Alta",
                    "Abierto"
                ),
                (
                    "Solicitud de acceso",
                    "Se necesita habilitar el acceso al sistema.",
                    "Media",
                    "En proceso"
                ),
                (
                    "Actualizar información",
                    "Corregir los datos de un registro.",
                    "Baja",
                    "Cerrado"
                )
            ])


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["email"] = user["email"]
            return redirect(url_for("dashboard"))

        flash("Correo o contraseña incorrectos.", "error")

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        priority = request.form.get("priority", "Media")

        if priority not in ("Baja", "Media", "Alta"):
            priority = "Media"

        if not title or not description:
            flash("Completa el título y la descripción.", "error")
        else:
            with get_db() as db:
                db.execute("""
                    INSERT INTO tickets
                        (title, description, priority, status, created_by)
                    VALUES (?, ?, ?, 'Abierto', ?)
                """, (
                    title, description, priority, session["user_id"]
                ))

            flash("Ticket creado correctamente.", "success")

        return redirect(url_for("dashboard"))

    with get_db() as db:
        tickets = db.execute("""
            SELECT * FROM tickets ORDER BY id DESC
        """).fetchall()

    return render_template(
        "dashboard.html",
        tickets=tickets,
        email=session.get("email")
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Crea la base de datos y las tablas al iniciar la aplicación.
init_db()


if __name__ == "__main__":
    app.run(debug=True)