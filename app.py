import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
from openai import AzureOpenAI

# Load environment variables
load_dotenv(override=True)

# Create AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.getenv("AZURE_OPENAI_KEY", ""),
    api_version="2024-02-15-preview"
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Validate and fetch environment variables
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not azure_endpoint or not api_key or not deployment_name:
    raise ValueError("AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME must be set as environment variables.")

openai.api_type = "azure"
openai.api_base = azure_endpoint
openai.api_version = "2024-02-15-preview"
openai.api_key = api_key

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def perform_query_chat(message_history):
    response = client.chat.completions.create(
        model="gpt35turbo16k",
        messages=message_history,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response

@app.route('/')
def index():
    return render_template('test.html')

message_history = []

@app.route('/chat', methods=['POST'])
def chatbot():
    user_message = request.form.get('user_message')

    if user_message:
        prompt = f"""
         As a QA Engineer, your task is to Acceptance Criteria enclosed within curly braces.And each test case should contain the following mentioned details:
         1. Test Case Name
         2. Test Step
         3. Expected Result
        {user_message}
            
       
        """

        message_history.append({"role": "user", "content": prompt})
        response = perform_query_chat(message_history)

        # Correctly access the response content
        assistant_message = response.choices[0].message.content

        message_history.append({"role": "assistant", "content": assistant_message})

        return jsonify({"response": assistant_message})
    return jsonify({"response": "Please provide a message!"})

if __name__ == '__main__':
    app.run(debug=False)  # Debug mode should be False in production
