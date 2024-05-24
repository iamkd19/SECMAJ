from flask import Blueprint, render_template, request, redirect, url_for
import sqlite3

admin_bp = Blueprint('admin', __name__)

# Route to the admin panel
@admin_bp.route('/admin')
def admin():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM subjects")
    subjects = cursor.fetchall()
    conn.close()

    return render_template('admin.html', subjects=subjects)

# Route to handle adding new subjects
@admin_bp.route('/add_subject', methods=['POST'])
def add_subject():
    name = request.form['name']
    year = request.form['year']
    total_classes = request.form['total_classes']

    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO subjects (name, year, total_classes) VALUES (?, ?, ?)", (name, year, total_classes))
    conn.commit()
    conn.close()

    return redirect(url_for('admin.admin'))

# Route to handle deleting a subject
@admin_bp.route('/delete_subject/<int:id>')
def delete_subject(id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for('admin.admin'))
