####################################################################
import openai
import os

# Check the installed version of the OpenAI library
print(openai.__version__)

# Alternatively, you can determine the version via the command line:
# pip show openai

# Assign your OpenAI organization ID and API key to environment variables
os.environ['OPENAI_ORGANIZATION'] = "your_organization_id"
os.environ['OPENAI_API_KEY'] = "your_api_key"

# Retrieve and use the API key from environment variables
openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = os.getenv('OPENAI_API_KEY')

#######################OpenAI API: Complete model#########################
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# directly use openai API
completion = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt = 'how many prime number between 1 and 100?',
  max_tokens=7,
  temperature=0
)
print(completion.choices[0].text)

#######################OpenAI API: streaming interactions#######################

client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Can you tell me how to do EDA?"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

##################OpenAI's API  for data analysis,###############################
df = pd.read_csv('production_sale.csv')

data_str = str(df.dtypes)
prompt = " given pandas data frame : production_sale with the columns: " + \
           data_str + \
           ",where the data frame is read from \
           C:/data/production_sale.csv   "
           
question = " provide me the python code for computing and print \
            Peason correlation between product and sales,\
            and plot side by side  bar chart such as \
            df.plot(x='month_seq', y=['product', 'sales'], kind='bar') \
            Note, only provide me the pure code without.comment "

completion = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt = prompt + question,
  max_tokens=180,
  temperature=0
)
cd = completion.choices[0].text
exec(cd)      

####################OpenAI API:text-embedding-ada-002 for text embedding#################
from openai import OpenAI
client = OpenAI()

# define text embedding function 
def get_embedding(text_str):
   text_str = text_str.replace("\n", " ")
   response = client.embeddings.create(input = [text_str], model='text-embedding-ada-002').
   em_vector = response.data[0].embedding
   return em_vector

my_question = 'How to pay off my credit card debt early'

embedding = get_embedding(my_question)

df = pd.read_csv('creditcard.csv')
df['ada_question'] = df['question'].apply(lambda x: get_embedding(x))
df.to_csv('creditcard_em.csv', index = False)

import ast
DF = pd.read_csv('creditcard_em.csv')
DF['ada_question'] = DF['ada_question'].apply(ast.literal_eval)

def vector_similarity(vector1, vector2):
    a = np.array(vector1)
    b = np.array(vector2)
    similarity_score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 
    return similarity_score

def get_vector_similarity(row, embedding, column_name):
    ada_question_embedding = row[column_name] 
    return vector_similarity(ada_question_embedding, embedding)

DF['similarities'] = DF.apply(get_vector_similarity, axis=1, args=(embedding, 'ada_question'))
DF['similarities'].idxmax()

find_question = DF.loc[dff['similarities'].idxmax()]['question']
find_answer = DF.loc[dff['similarities'].idxmax()]['answer']
distance = DF.loc[dff['similarities'].idxmax()]['similarities']

#######################usage of AzureOpenAI############################

from openai import AzureOpenAI as AzureOpenAI0
client = AzureOpenAI0(
  api_key = OPENAI_API_KEY,  
  api_version = "2023-08-01-preview",
  azure_endpoint = OPENAI_API_BASE
)

q = '"one is 3, two is 3, three is 5, then six is"'
response = client.chat.completions.create(
    model='gpt432k',
    messages=[
        {"role": "system", "content": " You are a smart guy"},
        {"role": "user", "content": q}
    ]
)

print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)

###################Use Mistral LLM#########################################

from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(" Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)

q = """ write python code based on the following instruction: 
utilizes the numpy library  to simulate daily stock prices over a period of 50 days. The stock price for each day should be generated using  a normal distribution with a mean that increases linearly by 0.05*t from an initial value of 30 and a constant standard deviation of 3. Here, t represents the day sequence number starting from 1 to 50. After generating these stock prices,  use matplotlib's plt.plot function to plot the daily stock prices, setting t as the x-axis and the generated  stock prices as the y-axis.  """     

print(llm(q))

########################example: LangChain with Azure OpenAI############################
from langchain_openai import AzureChatOpenAI
question = "Please provide me an brief answer of the following question, \
                 can I use linear regression if the Y is not normally distributed?"

model = AzureChatOpenAI(
azure_endpoint=OPENAI_API_BASE,
openai_api_version="2023-08-01-preview",
azure_deployment='gpt432k',
openai_api_key=OPENAI_API_KEY,
openai_api_type="azure",
)

result = model.predict(question)
print (result)

###################LangChain with Azure OpenAI: conversational manner##############
from langchain.schema import HumanMessage, SystemMessage, AIMessage

model = AzureChatOpenAI(
azure_endpoint=OPENAI_API_BASE,
openai_api_version="2023-08-01-preview",
azure_deployment='gpt432k',
openai_api_key=OPENAI_API_KEY,
openai_api_type="azure",
)

q = "what about car?"
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="who invented telephone?"),
    AIMessage(content="Alexander Graham Bell"),
    HumanMessage(content= q),    
]

# Concatenate the contents of the messages, response = model([message])
combined_message = " ".join([msg.content for msg in messages])

response1 = model.predict(combined_message)
response2 = model(messages).content

########################show LangChain's dynamic prompts ###########################

## necessary LangChain lib
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import (
StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

template = """
I need your expertise to solve some data problems

Here is a question:

are there any outliers in [1.8, 2.5, 10.3, 8.1, 0.2, 5.8, 32.3, 8.1, 12.5, 9.5, 0, 15.7] ?
use quantile way, 

Yes, there is an outlier in the dataset:32.3 This was identified using the 
quantile (IQR) method 

What are outliers for the list {num} ?
"""

## set up PromptTemplate
prompt = PromptTemplate(input_variables=["num"],template=template,)
## new question
num_lst = [11.5, 0.5, 29.8, 102.6, 10.7, 400.5, 22, 55, 78.6, 9, 201, 97.3, -12]

print(model.predict(prompt.format(num = num_lst, )))

chain = LLMChain(llm=model, prompt=prompt)
# Run the chain only specifying the input variable.
print(chain.run(num = num_lst))

################integrate LangChain with the Mistral LLM LLM###############

from langchain.llms import CTransformers as CTransformers_lc

llm_mis = CTransformers_lc(model = r"C:\Users\me\Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf")

chain_mis = LLMChain(llm=llm_mis, prompt=prompt)
print(chain_mis.run(num = num_lst)) 

#############ConversationChain and ConversationBufferMemory#############################

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# retain memory throughout the conversation: setup memory 
memory = ConversationBufferMemory(return_messages=True)
# initializing memory
memory.save_context({"input": "we talk about lending business"}, {"ouput": "sure, I am an expert in credit card business"}) #initializing the conversation

# see what is in the memory 
memory.load_memory_variables({})

# modelgpt4 is an Azure OpenAI API 
conversation = ConversationChain(
    llm = modelgpt4,   verbose=True,  memory=ConversationBufferMemory()
)

# ask AI bot answer the first question
question1 = 'what is the 30 days delinquency?'
response1  = conversation.predict(input = question1)

# second question based on the context and same conversation instance
question2 = 'then what about 90 days and impact on me?'
conversation.predict(input = question2)


######application of LangChain chains to address a data science question##############

question_prompt = PromptTemplate.from_template(
    """You are a data science expert. Given the user's request , 
       it is your job to summarize the main points.
       User's request: {request}
       Summary:"""
)

# Template for a response
response_prompt = PromptTemplate.from_template(
    """You are a data science expert. Given the summary user's request, 
        it is your job to write a professional response.
request Summary:
{summary}
 Response:"""
)

question_chain = question_prompt | llm | StrOutputParser()
response_chain = (
    {"summary": question_chain}
    | response_prompt
    | llm
    | StrOutputParser()
)

# Using the response_chain chain with  user's request

q = """ I am developing predictive models using variables x1 (profit) 
and x2 (cost) to forecast the binary outcome y (good or bad). When I employ either x1 alone  or x2 alone for prediction, the performance of the model is notably poor.  However, when I incorporate both x1 and x2 into the model, 
 the performance significantly improves, despite the fact that x1 and x2
 have a high correlation of 95% leading to Multicollinearity.
 This situation has left me puzzled. """
      
response_chain.invoke(
  {"request": q}
)

######employ the ConversationalRetrievalChain and AzureChatOpenAI LLM for querying based on documents##

from text_loader import TextLoader
from text_splitter import CharacterTextSplitter
from chroma import Chroma
from azure_openai_embeddings import AzureOpenAIEmbeddings

loader = TextLoader("guide_data.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=AzureOpenAIEmbeddings(deployment='textembeddingada002',
                                    model='text-embedding-ada-002',
                                    azure_endpoint=openai.api_base,
                                    openai_api_key=openai.api_key,
                                    openai_api_type="azure"),
persist_directory="C:/data_st/chroma_db")

# Next, define the embeddings model and a function to retrieve data:
embeddings_model = AzureOpenAIEmbeddings(deployment='textembeddingada002',
                                   model='text-embedding-ada-002',
                                   azure_endpoint=openai.api_base,
                                   openai_api_type="azure")
def get_retriever():
    loaded_vectordb = Chroma(persist_directory="C: /data_st/chroma_db", embedding_function=embeddings_model)
    retriever = loaded_vectordb.as_retriever()
return retriever

Set up the conversation history as follows:

conversation_history = [
    SystemMessage(content="You are a helpful assistant skilled in using Python for data analysis and predictive modeling."),
    HumanMessage(content="I need to analyze data and build predictive models."),
    AIMessage(content="Certainly, I can assist."),
]

#Define message templates for the system and user:
general_system_template = r""" 
Given a specific context, please give a short answer less than 100 words to the question. If you don't know the answer, just say that you don't know """ 

general_system_template = general_system_template + \
""" If you cannot find context in conversation history, just give general answer
 ----
{context}
---- """

general_user_template = "Question:```{question}```"

messages = [            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

# Initialize the AzureChatOpenAI model and set up the Conversational Retrieval Chain:

llm = AzureChatOpenAI(azure_endpoint=openai.api_base,
                      openai_api_version="2023-08-01-preview",
                      azure_deployment='gpt432k',
                      openai_api_key=openai.api_key,
                      openai_api_type="azure",
                    )
    
qa = ConversationalRetrievalChain.from_llm(
            llm= llm,
            retriever=get_retriever(),
            chain_type="stuff",
            combine_docs_chain_kwargs={'prompt': qa_prompt}
        ) 

answer = qa({"question": "tell me the business background of company", "chat_history": conversation_history})


####Use the ConversationChain with the 'Passing data through' method (RunnablePassthrough)#####

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. """
hint = """ Use the following pieces of retrieved context to answer the question. 
  If you don't know the answer, just say that you don't know. 
  Use three sentences maximum and keep the answer concise.
  Question: {question} 
  Context: {context} 
  Answer:
  """
template = template +  hint 
    
prompt = ChatPromptTemplate.from_template(template)
  
llm = AzureChatOpenAI(azure_endpoint=openai.api_base,
                      openai_api_version="2023-08-01-preview",
                      azure_deployment='gpt4',
                      openai_api_key=openai.api_key,
                      openai_api_type="azure")
                      
retriever = get_retriever()

question =  "tell me the business background of company"
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

response = rag_chain.invoke(question)

################Show Maximum Marginal Relevance (MMR) search in RAG#################

from langchain.llms import OpenAI as OpenAI_llms

template = """You are an assistant for question-answering tasks. """

def get_chunks(question,k):
  loaded_vectordb = Chroma(persist_directory= "C:/ data_st/chroma_db", embedding_function=embeddings_model)
  docs = loaded_vectordb.max_marginal_relevance_search(question)
  chunks = ' '.join([chunk.page_content for chunk in docs])
  return chunks

question =  "tell me the business background of company based on the context info"

retrieved = get_chunks(question,5)
hint = """ Use the following pieces of retrieved context to answer the question. 
  If you don't know the answer, just say that you don't know. 
  Use up to 5 sentences maximum and keep the answer concise.
  use the following the context: "{retrieved}
  Answer the question: {question} 
  """

prompt = template +  hint 
llm = OpenAI_llms(temperature = 0.03)
response = llm.predict(prompt)

#############LangChain agent: customer tool for data analysis#######################
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

datapath = 'test_data.csv'
def loaddata(datapath: str) -> pd.DataFrame():
    """load data """
    import pandas as pd
    data = pd.read_csv(datapath)    
    return data

def cor(a: str, b: str) -> float:
    """calculate correlation coefficient  between two variables."""
    return data[a].corr(data[b])

load_tool = StructuredTool.from_function(loaddata)
cor_tool = StructuredTool.from_function(cor)

cal_agent = initialize_agent([load_tool, cor_tool], llm, 
           agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True                
           )
     
cal_agent.run({"input": "load data from " + datapath + ", and calculate correlation between 'x' and 'y' columns?"})

#################LangChain agent: YahooFinanceNewsTool####################################

from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
import yfinance

tools = [YahooFinanceNewsTool()]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run(
    "What is the most important news for US stock market today?",
)

tool = YahooFinanceNewsTool()
tool.run("NET")

###########LangChain agent: Python code generation or 'Python REPL' #######

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python \
        commands. Input should be a valid python command. \
            If you want to see the output of a value, \
                you should print it out with `print(...)`.",
                
    func=python_repl.run,
)

agent = initialize_agent([repl_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run("What is the sampled list \
                     from the list range(1, 100) if I randomly draw 5 number?")



