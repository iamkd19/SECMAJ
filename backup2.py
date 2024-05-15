from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Attendance App!"

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, entry_time, exit_time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return jsonify({'error': 'No attendance data available for the selected date'}), 404
    
    return jsonify({'attendance_data': attendance_data})

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request
# import sqlite3
# from datetime import datetime

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html', selected_date='', no_data=False)

# @app.route('/attendance', methods=['POST'])
# def attendance():
#     selected_date = request.form.get('selected_date')
#     selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
#     formatted_date = selected_date_obj.strftime('%Y-%m-%d')

#     conn = sqlite3.connect('attendance.db')
#     cursor = conn.cursor()

#     cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
#     attendance_data = cursor.fetchall()

#     conn.close()

#     if not attendance_data:
#         return render_template('index.html', selected_date=selected_date, no_data=True)
    
#     return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)

# if __name__ == '__main__':
#     app.run(debug=True)
