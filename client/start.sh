  #!/bin/bash

  # Start the Flask server
  python client.py &
  FLASK_PID=$!

  sleep 10

  for i in {1..20}; do
    echo "Starting training round $i"
    response=$(curl -s http://localhost:5000/train)
    
    if echo $response | grep -q "Early stopping triggered"; then
      echo "Early stopping detected. Terminating training."
      break
    fi
    
    # Query server status
    server_status=$(curl -s http://central-server:5000/status)
    if echo $server_status | grep -q "\"early_stopping\":true"; then
      echo "Early stopping detected from server. Terminating training."
      break
    fi
    
    SLEEP_TIME=$((5 + RANDOM % 15))
    echo "Waiting $SLEEP_TIME seconds until next round..."
    sleep $SLEEP_TIME
  done

  wait $FLASK_PID
  ```