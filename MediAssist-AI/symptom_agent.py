# since we wil be using the LLm interference we will be using the package langchain-azure-ai from https://python.langchain.com/docs/integrations/providers/microsoft/
import getpass #The getpass module is used to safely input passwords or sensitive information from the user without displaying them on the screen (no echo).
import os # allows to interact with the operating system

#checking if the key is available in envornment
if not os.getenv("AZURE_INFERENCE_CREDENTIAL"):
    os.environ ["AZURE_INFERENCE_CREDENTIAL"]== getpass.getpass("Enter your Azure API key: ")

#ckecking if the endpoint is available in the enviornment
if not os.getenv("AZURE_INFERENCE_ENDPOINT"):
    os.environ ["AZURE_INFERENCE_ENDPOINT"]== getpass.getpass("Enter your model: ")
