from flask import Flask, render_template, request
import sqlite3
from datetime import datetime, timedelta

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

# Function to calculate and retrieve average attendance over the last 30 days
def get_average_attendance():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Get today's date and the date 30 days ago
    today = datetime.now()
    last_30_days = today - timedelta(days=30)
    formatted_today = today.strftime('%Y-%m-%d')
    formatted_last_30_days = last_30_days.strftime('%Y-%m-%d')

    query = """
    SELECT name, COUNT(*) as total_days,
           ROUND((COUNT(*) * 100.0) / 30, 2) as average_attendance_percentage
    FROM attendance
    WHERE date BETWEEN ? AND ?
    GROUP BY name;
    """
    cursor.execute(query, (formatted_last_30_days, formatted_today))
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
