import streamlit as st
import pandas as pd
import os
import pickle
import time
import together
import logging

# Load the API key from environment variables 
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if TOGETHER_API_KEY is None:
    raise Exception("API key not found. Please set the TOGETHER_API_KEY environment variable.")

class TogetherLLM:
    """Together large language models with retry mechanism."""
    def __init__(self, model: str, api_key: str, temperature: float = 0.7, max_tokens: int = 120000, retries: int = 3, backoff_factor: float = 1.5):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff_factor = backoff_factor
        together.api_key = api_key  # Set the API key directly

    def call(self, prompt: str) -> str:
        """Call to Together endpoint with retry mechanism."""
        attempt = 0
        while attempt < self.retries:
            try:
                response = together.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                # Return response if successful
                return response.choices[0].text.strip()
            except (together.error.APIError, together.error.APIConnectionError, together.error.InvalidRequestError) as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                time.sleep(self.backoff_factor ** attempt)  # Exponential backoff

        # If all retries fail, raise an exception
        raise Exception(f"Failed to complete after {self.retries} retries")

def load_csv_file(file) -> pd.DataFrame:
    """Load the CSV file into a DataFrame."""
    return pd.read_csv(file)

def process_chunk_sequentially(chunk: pd.DataFrame, model: TogetherLLM, previous_summary: str = "") -> str:
    """Process a chunk sequentially and return the biography section."""
    chunk['Transkript'] = chunk['Transkript'].astype(str)
    chunk_text = chunk['Transkript']

    if previous_summary:
        prompt = (
            "Hier ist die bisher erstellte Biografie:\n{previous_summary}\n\n"
            "Bitte fassen Sie die folgenden Interviewinformationen zusammen und konzentrieren Sie sich auf wichtige Lebensereignisse, "
            "insbesondere auf die erwähnten Jahre und biografisch bedeutsame Ereignisse. Vermeiden Sie Wiederholungen und irrelevante Details. "
            "Integrieren Sie neue relevante Informationen in die bestehende Biografie.\n\n{chunk_text}"
        )
    else:
        prompt = (
            "Erstellen Sie eine vollständige und detaillierte Biografie basierend auf den folgenden Interviewinformationen. "
            "Konzentrieren Sie sich dabei auf wichtige Lebensereignisse und die in den Interviews erwähnten Jahre. "
            "Vermeiden Sie Wiederholungen und irrelevante Details.\n\n{chunk_text}"
        )
    
    biography_section = model.call(prompt.format(previous_summary=previous_summary, chunk_text=chunk_text))
    return biography_section

def main():
    st.title("Biography Generator using Together LLM")

    st.write("Please upload a csv file containing the transcripts:")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file
        df = load_csv_file(uploaded_file)

        # Initialize the LLM model
        model = TogetherLLM(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            api_key=TOGETHER_API_KEY,
            temperature=0.7,
            max_tokens=120000,
            retries=5,
            backoff_factor=1.5
        )

        # Process the chunks
        chunk_size = 250
        previous_summary = ""

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            try:
                previous_summary = process_chunk_sequentially(chunk, model, previous_summary)
            except Exception as e:
                st.error(f"Error processing chunk: {e}")

        st.subheader("Generated Summary")
        st.write(previous_summary)

        # Finalize the complete biography
        final_prompt = (
            "Hier ist die vollständige Biografie, die aus verschiedenen Abschnitten des Interviews extrahiert und kombiniert wurde. "
            "Bitte verfeinern Sie diese Biografie, indem Sie sie flüssiger und kohärenter machen, Redundanzen entfernen und sicherstellen, "
            "dass die Erzählung chronologisch und vollständig ist.\n\n{final_biography}"
        )

        final_biography = model.call(final_prompt.format(final_biography=previous_summary))

        st.subheader("Generated Biography")
        st.write(final_biography)

if __name__ == "__main__":
    main()
