# Multiagent Workflow with GPT-4 Integrated Streamlit App

## Introduction

This project integrates a multiagent workflow system with GPT-4, showcased through a Streamlit app. It combines technologies like Matplotlib, NetworkX, and OpenAI's GPT-4 to process user queries, generating and evaluating Python code accordingly. The system is designed to improve over time through a feedback mechanism that adapts based on user interactions.

## Installation

Ensure Python 3.6+ is installed, then follow these steps:

1. Clone the repository and navigate to the project directory.
2. Install dependencies with `pip install -r requirements.txt`.

## Usage

Launch the Streamlit app by executing `streamlit run app.py` in your terminal and follow the on-screen instructions.

## Features

- State Graph Workflow for managing processing nodes.
- GPT-4 integration for code generation and evaluation.
- Custom vector storage for efficient query-code association.
- Streamlit web interface for easy interaction.
- User feedback loop for continuous system improvement.

## Dependencies

Key dependencies include streamlit, matplotlib, networkx, numpy, and openai. See `requirements.txt` for a full list.

## Configuration

Set your OpenAI API key in the environment:

```bash
export OPENAI_API_KEY='your_api_key_here'
