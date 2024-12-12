import os
import json
import google.generativeai as genai  # Google Gemini API Client
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List, Any, Dict

# Set up Google Gemini API Key
os.environ['GOOGLE_API_KEY'] = "API_KEY"  # Replace with your Gemini API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Corrected LangChain wrapper for Google Gemini
class GoogleGeminiLLM(BaseLLM):
    """Custom LangChain LLM wrapper for Google Gemini."""

    model_name: str = "gemini-pro"  # Default model name for Gemini
    temperature: float = 0.7  # Optional: Control creativity of the response

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """Generate responses for a list of prompts."""
        try:
            model = genai.GenerativeModel(self.model_name)
            generations = []
            for prompt in prompts:
                response = model.generate_content(prompt)
                output = response.text if response else "No response received."
                generations.append([Generation(text=output)])  # LangChain-compliant response
            return LLMResult(generations=generations)
        except Exception as e:
            return LLMResult(generations=[[Generation(text=f"Error: {str(e)}")]])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "Google Gemini"

# Initialize Google Gemini LLM
llm = GoogleGeminiLLM()

# Prompt Template for MCQ Generation
TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

# Prompt Template for Quiz Evaluation
TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
If the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

# Define Chains
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# Sequential Chain for Generating and Evaluating Quiz
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)
