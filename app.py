from flask import Flask, render_template, request, redirect, url_for
import json
import os
import subprocess
import shutil

app = Flask(__name__)

# Default values for the speeds
max_speed_permitted = 130
high_priority_speed = 130
medium_priority_speed = 100
low_priority_speed = 80

@app.route('/')
def index():
    # Read the optimized schedules safely
    try:
        with open('optimized_schedules.json', 'r') as f:
            schedules = json.load(f)
    except FileNotFoundError:
        schedules = {}

    # Render the page with the current schedules and input values
    return render_template('index.html', schedules=schedules)

@app.route('/update', methods=['POST'])
def update():
    global max_speed_permitted, high_priority_speed, medium_priority_speed, low_priority_speed

    # Get updated values from the form
    max_speed_permitted = int(request.form['max_speed'])
    high_priority_speed = int(request.form['high_priority_speed'])
    medium_priority_speed = int(request.form['medium_priority_speed'])
    low_priority_speed = int(request.form['low_priority_speed'])

    # Set the environment variables for THIRD.py
    os.environ['MAX_SPEED'] = str(max_speed_permitted)
    os.environ['HIGH_PRIORITY_SPEED'] = str(high_priority_speed)
    os.environ['MEDIUM_PRIORITY_SPEED'] = str(medium_priority_speed)
    os.environ['LOW_PRIORITY_SPEED'] = str(low_priority_speed)

    # Run THIRD.py and check for errors
    try:
        # Running THIRD.py and capturing output
        result = subprocess.run(['python', 'THIRD.py'], capture_output=True, text=True, check=True)

        # Output from THIRD.py to monitor execution
        print(result.stdout)

        # Move the generated image file to the static folder
        if os.path.exists("average_fragment_speed_distribution.png"):
            print("Image file exists and is being moved to static folder.")
            shutil.move("average_fragment_speed_distribution.png", "static/average_fragment_speed_distribution.png")
        else:
            print("Error: Plot file not generated.")
        
    except subprocess.CalledProcessError as e:
        # Log any errors in the subprocess call
        print(f"Error in THIRD.py: {e.stderr}")
        return redirect(url_for('index'))

    # Ensure the file is updated (overwritten) and not deleted
    try:
        with open('optimized_schedules.json', 'r') as f:
            schedules = json.load(f)
    except FileNotFoundError:
        schedules = {}

    # Render the updated schedules and plot
    return render_template('index.html', schedules=schedules)

if __name__ == '__main__':
    app.run(debug=True)
