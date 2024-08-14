import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class LLM:
    def __init__(self) -> None:
        """
        Initialize the OpenAI API
        """
        pass

    def create_llm(self, model_name: str = 'gpt-3.5-turbo') -> None:
        """
        Create an instance of the OpenAI API

        Parameters:
        -----------
        model_name: str, optional
            The name of the language model to use. Default is 'gpt-3.5-turbo'
        """
        self.llm = ChatOpenAI(model_name=model_name, api_key=os.getenv('OPENAI_API_KEY'))

    