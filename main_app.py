import os
import PyPDF2
import json
import traceback
import pandas as pd
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQ_GEN import generate_evaluate_chain
from src.mcqgenerator.logger import logging


# Load JSON file
with open(r'C:\Users\DELL\MCQ_GENERATOR\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Create a title for the app
st.title("MCQs Creator Application with LangChain")

# Create a form using st.form
with st.form("user_inputs"):
    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF or .txt file")

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")

    # Add Button
    button = st.form_submit_button("Create MCQs")

# Check if the button is clicked and all fields have input
if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Loading..."):
        try:
            text = read_file(uploaded_file)  # Function to read file content
            
            # Count tokens and estimate API call cost
            response = generate_evaluate_chain(
                {
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON)
                }
            )
            
            # Uncomment to display response
            # st.write(response)  

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")

        else:
            # After calling the chain
            if isinstance(response, dict):
             
                quiz = response.get("quiz", None)
                quiz = quiz.replace("### RESPONSE_JSON", "").strip()  # Remove any non-JSON prefix
                if quiz:
                    try:
                        # Get table data from the quiz

                        table_data = get_table_data(quiz)
                        if table_data:
                            # Create a DataFrame and display it as a table
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1  # To start the index from 1
                            st.table(df)
                            # Display review text
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in table data format.")
                    except Exception as e:
                        st.error(f"Error processing table data: {e}")
                else:
                    st.error("Quiz data is missing in the response.")
            else:
                st.write(response)
