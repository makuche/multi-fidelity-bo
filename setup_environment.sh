#!/bin/bash

# If virtual environment folder does not exist yet, create it
if [ ! -d "venv" ]; then
    # Check if virtualenv is installed on the system
    if [ -x "$(command -v virtualenv)" ]; then
        # Create a new virtualenv
        virtualenv venv
        source venv/bin/activate
        pip3 install -r requirements.txt
        deactivate
    fi
fi