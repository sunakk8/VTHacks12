from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd
from main import train

app = Flask(__name__)
app.secret_key = "super secret key"


@app.route('/')
def home():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    name = request.form.get('input1')
    device_type = request.form.get('devices')
    platform = request.form.get('platforms')
    time_spent = request.form.get('input3')
    watch_reason = request.form.get('reason')
    addiction_level = request.form.get('input5')

    # Store data in the session
    session['name'] = name
    session['device_type'] = device_type
    session['platform'] = platform
    session['time_spent'] = time_spent
    session['watch_reason'] = watch_reason
    session['addiction_level'] = addiction_level


    # Process data (here we simply print it, but you can modify this to save it to a file or database)
    print(f"Name: {name}")
    print(f"Most Used Device Type: {device_type}")
    print(f"Platform: {platform}")
    print(f"Total Time Spent: {time_spent}")
    print(f"Watch Reason: {watch_reason}")
    print(f"Addiction Level: {addiction_level}")

    model, input = train(np.array([platform,device_type,time_spent, 
                               watch_reason,addiction_level, 4]))
    final = input.to_frame()
    prediction = model.predict(final.T)
    # Render a result template or redirect to a thank you page
    return render_template('thank_you.html', name=name, device_type=device_type, platform=platform, time_spent=time_spent, watch_reason=watch_reason, addiction_level=addiction_level, productivity_level=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)





