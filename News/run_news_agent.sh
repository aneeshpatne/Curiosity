#!/bin/bash
cd /home/aneesh/Desktop/Code/Curiosity/News  # Ensure the working directory is correct
source .venv/bin/activate  # Activate the virtual environment
python news-agent.py  # Run the script
deactivate  # Deactivate the virtual environment (optional)
