from flask import Flask, render_template, request
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Route to the index page
@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

# Route to handle attendance data for a specific date
@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, entrytime, exittime FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

# Function to calculate and retrieve average attendance
def get_average_attendance():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    query = """
    SELECT name, COUNT(*) as total_days,
           ROUND(COUNT(*) / (SELECT COUNT(DISTINCT date) FROM attendance), 2) as average_attendance
    FROM attendance
    GROUP BY name;
    """
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

# Route to handle the request for average attendance
@app.route('/average_attendance')
def average_attendance():
    avg_attendance_data = get_average_attendance()
    return render_template('average_attendance.html', avg_attendance_data=avg_attendance_data)

if __name__ == '__main__':
    app.run(debug=True)
