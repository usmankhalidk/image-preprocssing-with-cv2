#!/bin/bash

# Set ngrok authtoken
ngrok config add-authtoken $NGROK_AUTHTOKEN

# Start the Flask application in the background
python app.py &

# Wait a moment for Flask to start
sleep 5

# Start ngrok to expose the Flask app
ngrok http 3000 --log stdout
