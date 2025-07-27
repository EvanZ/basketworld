# BasketWorld Interactive Web App

This directory contains the source code for a web-based, interactive version of the BasketWorld simulation. It allows a human user to play against the trained AI agents.

## Directory Structure

- `/backend`: A Python FastAPI server that runs the core simulation, loads the AI models, and exposes an API for the frontend.
- `/frontend`: A Vue.js single-page application that provides the user interface, renders the game state, and communicates with the backend.

## Getting Started (High-Level)

1.  **Run the Backend:** Navigate to `app/backend`, install dependencies from `requirements.txt`, and run the server with `uvicorn main:app --reload`.
2.  **Run the Frontend:** Navigate to `app/frontend`, install dependencies with `npm install`, and run the development server with `npm run serve`.
3.  **Play:** Open the frontend URL in your browser. 