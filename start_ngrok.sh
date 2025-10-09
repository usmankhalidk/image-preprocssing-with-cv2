#!/bin/bash

# Set ngrok authtoken
export NGROK_AUTHTOKEN="31B81lOcitORGjrkLslRfWJRGiS_5cS11ehDoAw9Cv4Xrs7kQ"

# Start the Flask application in the background
echo "Starting Flask API..."
source venv/bin/activate
python app.py &
FLASK_PID=$!

# Wait for Flask to start
echo "Waiting for Flask to start..."
sleep 5

# Check if Flask is running
if ! curl -s http://localhost:3000/health > /dev/null; then
    echo "❌ Flask API failed to start"
    exit 1
fi

echo "✅ Flask API is running on localhost:3000"

# Start ngrok
echo "Starting ngrok tunnel..."
ngrok http 3000 --log stdout
