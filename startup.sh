#!/bin/bash
pip install -r requirements.txt
streamlit run Home.py --server.port=$PORT --server.enableCORS false