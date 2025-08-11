# since we wil be using the LLm interference we will be using the package langchain-azure-ai from https://python.langchain.com/docs/integrations/providers/microsoft/
import getpass #The getpass module is used to safely input passwords or sensitive information from the user without displaying them on the screen (no echo).
import os # allows to interact with the operating system

#checking if the key is available in envornment
if not os.getenv("AZURE_INFERENCE_CREDENTIAL"):
    os.environ ["AZURE_INFERENCE_CREDENTIAL"]= getpass.getpass("Enter your Azure API key: ")

#ckecking if the endpoint is available in the enviornment
if not os.getenv("AZURE_INFERENCE_ENDPOINT"):
    os.environ ["AZURE_INFERENCE_ENDPOINT"]= getpass.getpass("Enter your model: ")

#istantiation
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
assist = AzureAIChatCompletionsModel(
    model_name="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#using Lagchain to gather symptoms
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are a medical proffesional. You task is to have a converstaion with the user to gather the information \
        regarding their symptoms, and then briefly summarise user's symptoms without lossing any information.\
        The summmary of information should be in format such that I can use it in embedding.\n\
        The users symptoms are {input}"
        ),
        ("user","{input}"),
    ]
)

#using chain
summary_sym = prompt | assist
symptoms = summary_sym.invoke (
    {"input": input("Please tell me your symptoms:\n")}
)
print (symptoms)