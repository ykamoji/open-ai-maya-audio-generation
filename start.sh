#!/bin/bash

set -e
echo "Checking everything is ready"
python3 init.py
echo "Starting voice generation"
python3 voiceGenerator.py
echo "Done"