import json
import boto3
import os
import time
import datetime
import PyPDF2
import csv
import re
import traceback
import base64
import operator

from botocore.config import Config
from io import BytesIO
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_aws import ChatBedrock
from multiprocessing import Process, Pipe

# from langchain_community.tools.tavily_search import TavilySearchResults
from PIL import Image

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from pydantic.v1 import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict, Literal, Sequence, Union
from langchain_aws import AmazonKnowledgeBasesRetriever
from tavily import TavilyClient  
from langgraph.constants import Send

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
debugMessageMode = os.environ.get('debugMessageMode', 'false')
projectName = os.environ.get('projectName')
LLM_for_chat = json.loads(os.environ.get('LLM_for_chat'))
LLM_for_multimodal= json.loads(os.environ.get('LLM_for_multimodal'))
selected_chat = 0
selected_multimodal = 0
useEnhancedSearch = False

knowledge_base_name = os.environ.get('knowledge_base_name')
    
multi_region_models = [   # claude sonnet 3.0
    {   
        "bedrock_region": "us-west-2", # Oregon
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "us-east-1", # N.Virginia
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "ca-central-1", # Canada
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "eu-west-2", # London
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    {
        "bedrock_region": "sa-east-1", # Sao Paulo
        "model_type": "claude3",
        "max_tokens": 4096,
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
    }
]
multi_region = 'disable'

reference_docs = []

secretsmanager = boto3.client('secretsmanager')
   
# api key to use LangSmith
langsmith_api_key = ""
try:
    get_langsmith_api_secret = secretsmanager.get_secret_value(
        SecretId=f"langsmithapikey-{projectName}"
    )
    # print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    secret = json.loads(get_langsmith_api_secret['SecretString'])
    #print('secret: ', secret)
    langsmith_api_key = secret['langsmith_api_key']
    langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
# api key to use Tavily Search
tavily_api_key = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    # print('secret: ', secret)
    tavily_api_key = json.loads(secret['tavily_api_key'])
    # print('tavily_api_key: ', tavily_api_key)
except Exception as e: 
    raise e

def check_tavily_secret(tavily_api_key):
    query = 'what is LangGraph'
    valid_keys = []
    for key in tavily_api_key:
        try:
            tavily_client = TavilyClient(api_key=key)
            response = tavily_client.search(query, max_results=1)
            # print('tavily response: ', response)
            
            if 'results' in response and len(response['results']):
                valid_keys.append(key)
        except Exception as e:
            print('Exception: ', e)
    # print('valid_keys: ', valid_keys)
    
    return valid_keys

tavily_api_key = check_tavily_secret(tavily_api_key)
# print('tavily_api_key: ', tavily_api_key)
print('The number of valid tavily api keys: ', len(tavily_api_key))

selected_tavily = -1
if len(tavily_api_key):
    os.environ["TAVILY_API_KEY"] = tavily_api_key[0]
    selected_tavily = 0
      
def tavily_search(query, k):
    global selected_tavily
    docs = []
        
    if selected_tavily != -1:
        selected_tavily = selected_tavily + 1
        if selected_tavily == len(tavily_api_key):
            selected_tavily = 0

        try:
            tavily_client = TavilyClient(api_key=tavily_api_key[selected_tavily])
            response = tavily_client.search(query, max_results=k)
            # print('tavily response: ', response)
            
            if "url" in r:
                url = r.get("url")
                
            for r in response["results"]:
                name = r.get("title")
                if name is None:
                    name = 'WWW'
            
                docs.append(
                    Document(
                        page_content=r.get("content"),
                        metadata={
                            'name': name,
                            'url': url,
                            'from': 'tavily'
                        },
                    )
                )   
        except Exception as e:
            print('Exception: ', e)
    return docs

# result = tavily_search('what is LangChain', 2)
# print('search result: ', result)

# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 

# Multi-LLM
def get_chat():
    global selected_chat
    
    if multi_region == 'enable':
        length_of_models = len(multi_region_models)
        profile = multi_region_models[selected_chat]
    else:
        length_of_models = len(LLM_for_chat)
        profile = LLM_for_chat[selected_chat]
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_chat = selected_chat + 1
    if selected_chat == length_of_models:
        selected_chat = 0
    
    return chat

def get_multi_region_chat(models, selected):
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    return chat

def get_multimodal():
    global selected_multimodal
    print('LLM_for_chat: ', LLM_for_chat)
    print('selected_multimodal: ', selected_multimodal)
        
    profile = LLM_for_multimodal[selected_multimodal]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    maxOutputTokens = 4096
    print(f'LLM: {selected_multimodal}, bedrock_region: {bedrock_region}, modelId: {modelId}')
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    multimodal = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
    
    selected_multimodal = selected_multimodal + 1
    if selected_multimodal == len(LLM_for_multimodal):
        selected_multimodal = 0
    
    return multimodal
    
# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary
    
def load_chatHistory(userId, allowTime, chat_memory):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text' and text and msg:
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg) 
                
def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
        
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content

    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")
    
    return msg

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
                                    
def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")        
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, multi_region_models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == len(multi_region_models):
            selected_chat = 0
    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()
    
    #print('filtered_docs: ', filtered_docs)
    return filtered_docs
    
def print_doc(doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content
            
    print(f"doc: {text}, metadata:{doc.metadata}")
    
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    if multi_region == 'enable':  # parallel processing
        print("start grading...")
        filtered_docs = grade_documents_using_parallel_processing(question, documents)

    else:
        # Score each doc    
        chat = get_chat()
        retrieval_grader = get_retrieval_grader(chat)
        for doc in documents:
            # print('doc: ', doc)
            print_doc(doc)
            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            print("score: ", score)
            
            grade = score.binary_score
            print("grade: ", grade)
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                continue
    
    global reference_docs 
    reference_docs += filtered_docs    
    # print('langth of reference_docs: ', len(reference_docs))
    
    # print('len(docments): ', len(filtered_docs))    
    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def get_answer_grader():
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )
        
    chat = get_chat()
    structured_llm_grade_answer = chat.with_structured_output(GradeAnswer)
        
    system = (
        "You are a grader assessing whether an answer addresses / resolves a question."
        "Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )
    answer_grader = answer_prompt | structured_llm_grade_answer
    return answer_grader
    
def get_hallucination_grader():    
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )
        
    system = (
        "You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts."
        "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts."
    )    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
        
    chat = get_chat()
    structured_llm_grade_hallucination = chat.with_structured_output(GradeHallucinations)
        
    hallucination_grader = hallucination_prompt | structured_llm_grade_hallucination
    return hallucination_grader
    
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


####################### LangGraph #######################
# Long form Writing Agent
#########################################################
    
# Workflow - Reflection
class ReflectionState(TypedDict):
    draft : str
    reflection : List[str]
    search_queries : List[str]
    revised_draft: str
    revision_number: int
        
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    advisable: str = Field(description="Critique of what is helpful for better writing")
    superfluous: str = Field(description="Critique of what is superfluous")

class Research(BaseModel):
    """Provide reflection and then follow up with search queries to improve the writing."""

    reflection: Reflection = Field(description="Your reflection on the initial writing.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current writing."
    )

class ReflectionKor(BaseModel):
    missing: str = Field(description="작성된 글에 있어야하는데 빠진 내용이나 단점")
    advisable: str = Field(description="더 좋은 글이 되기 위해 추가하여야 할 내용")
    superfluous: str = Field(description="글의 길이나 스타일에 대한 비평")

class ResearchKor(BaseModel):
    """글쓰기를 개선하기 위한 검색 쿼리를 제공합니다."""

    reflection: ReflectionKor = Field(description="작성된 글에 대한 평가")
    search_queries: list[str] = Field(
        description="현재 글과 관련된 3개 이내의 검색어"
    )
    
def reflect_node(state: ReflectionState):
    print("###### reflect ######")
    draft = state['draft']
    print('draft: ', draft)
    
    reflection = []
    search_queries = []
    for attempt in range(5):
        chat = get_chat()
        
        if isKorean(draft):
            structured_llm = chat.with_structured_output(ResearchKor, include_raw=True)
        else:
            structured_llm = chat.with_structured_output(Research, include_raw=True)
            
        info = structured_llm.invoke(draft)
        print(f'attempt: {attempt}, info: {info}')
                
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            # print('reflection: ', parsed_info.reflection)
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            search_queries = parsed_info.search_queries
                
            print('reflection: ', parsed_info.reflection)            
            print('search_queries: ', search_queries)     
        
            if isKorean(draft):
                translated_search = []
                for q in search_queries:
                    chat = get_chat()
                    if isKorean(q):
                        search = traslation(chat, q, "Korean", "English")
                    else:
                        search = traslation(chat, q, "English", "Korean")
                    translated_search.append(search)
                        
                print('translated_search: ', translated_search)
                search_queries += translated_search

            print('search_queries (mixed): ', search_queries)
            break
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "reflection": reflection,
        "search_queries": search_queries,
        "revision_number": revision_number + 1
    }

knowledge_base_id = None
def retrieve_from_knowledge_base(query):
    global knowledge_base_id
    if not knowledge_base_id:        
        client = boto3.client('bedrock-agent')         
        response = client.list_knowledge_bases(
            maxResults=10
        )
        print('response: ', response)
                
        if "knowledgeBaseSummaries" in response:
            summaries = response["knowledgeBaseSummaries"]
            for summary in summaries:
                if summary["name"] == knowledge_base_name:
                    knowledge_base_id = summary["knowledgeBaseId"]
                    print('knowledge_base_id: ', knowledge_base_id)
                    break
    
    relevant_docs = []
    if knowledge_base_id:    
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id, 
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}},
        )
        
        relevant_docs = retriever.invoke(query)
        print(relevant_docs)
    
    docs = []
    for i, document in enumerate(relevant_docs):
        print(f"{i}: {document.page_content}")
        if document.page_content:
            excerpt = document.page_content
        
        score = document.metadata["score"]
        print('score:', score)
        doc_prefix = "knowledge-base"
        
        link = ""
        if "s3Location" in document.metadata["location"]:
            link = document.metadata["location"]["s3Location"]["url"] if document.metadata["location"]["s3Location"]["url"] is not None else ""
            
            print('link:', link)    
            pos = link.find(f"/{doc_prefix}")
            name = link[pos+len(doc_prefix)+1:]
            encoded_name = parse.quote(name)
            print('name:', name)
            link = f"{path}{doc_prefix}{encoded_name}"
            
        elif "webLocation" in document.metadata["location"]:
            link = document.metadata["location"]["webLocation"]["url"] if document.metadata["location"]["webLocation"]["url"] is not None else ""
            name = "Web Crawler"

        print('link:', link)                    

        docs.append(
            Document(
                page_content=excerpt,
                metadata={
                    'name': name,
                    'url': link,
                    'from': 'RAG'
                },
            )
        )
    return docs
        
def revise_draft(state: ReflectionState):   
    print("###### revise_draft ######")
        
    draft = state['draft']
    search_queries = state['search_queries']
    reflection = state['reflection']
    print('draft: ', draft)
    print('search_queries: ', search_queries)
    print('reflection: ', reflection)
        
    if isKorean(draft):
        revise_template = (
            "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."                
            "draft을 critique과 information 사용하여 수정하십시오."
            "최종 결과는 한국어로 작성하고 <result> tag를 붙여주세요."
                            
            "<draft>"
            "{draft}"
            "</draft>"
                            
            "<critique>"
            "{reflection}"
            "</critique>"

            "<information>"
            "{content}"
            "</information>"
        )
    else:    
        revise_template = (
            "You are an excellent writing assistant." 
            "Revise this draft using the critique and additional information."
            # "Provide the final answer using Korean with <result> tag."
            "Provide the final answer with <result> tag."
                            
            "<draft>"
            "{draft}"
            "</draft>"
                        
            "<critique>"
            "{reflection}"
            "</critique>"

            "<information>"
            "{content}"
            "</information>"
        )
                    
    revise_prompt = ChatPromptTemplate([
        ('human', revise_template)
    ])
                              
    filtered_docs = []    
        
    # RAG - knowledge base
    if rag_state=='enable':
        for q in search_queries:
            docs = retrieve_from_knowledge_base(q)
            print(f'q: {q}, RAG: {docs}')
        
            if len(docs):
                filtered_docs += grade_documents(q, docs)
    
    # web search
    for q in search_queries:
        docs = tavily_search(q, 4)
        print(f'q: {q}, WEB: {docs}')
        
        if len(docs):
            filtered_docs += grade_documents(q, docs)
    
    """
    search = TavilySearchResults(max_results=4)
    for q in search_queries:
        response = search.invoke(q)
        print(f'q: {q}, WEB: {response}')
                
        docs = []
        for r in response:
            if 'content' in r:                        
                docs.append(
                    Document(
                        page_content=r.get("content"),
                        metadata={
                            'name': 'WWW',
                            'url': r.get("url"),
                            'from': 'tavily'
                        },
                    )
                )                
        print('docs from web search: ', docs)
        
        if len(docs):
            filtered_docs += grade_documents(q, docs)    
    """
    
    print('filtered_docs: ', filtered_docs)
              
    content = []   
    if len(filtered_docs):
        for d in filtered_docs:
            content.append(d.page_content)
        
    print('content: ', content)

    chat = get_chat()
    reflect = revise_prompt | chat
           
    res = reflect.invoke(
        {
            "draft": draft,
            "reflection": reflection,
            "content": content
        }
    )
    output = res.content
    # print('output: ', output)
        
    revised_draft = output[output.find('<result>')+8:len(output)-9]
    # print('revised_draft: ', revised_draft) 
            
    if revised_draft.find('#')!=-1 and revised_draft.find('#')!=0:
        revised_draft = revised_draft[revised_draft.find('#'):]

    print('--> draft: ', draft)
    print('--> reflection: ', reflection)
    print('--> revised_draft: ', revised_draft)
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
    return {
        "revised_draft": revised_draft,
        "revision_number": revision_number
    }
        
MAX_REVISIONS = 1
def should_continue(state: ReflectionState, config):
    print("###### should_continue ######")
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
    print("max_revisions: ", max_revisions)
            
    if state["revision_number"] > max_revisions:
        return "end"
    return "continue"
    
def buildReflection():
    workflow = StateGraph(ReflectionState)

    # Add nodes
    workflow.add_node("reflect_node", reflect_node)
    workflow.add_node("revise_draft", revise_draft)

    # Set entry point
    workflow.set_entry_point("reflect_node")
        
    workflow.add_conditional_edges(
        "revise_draft", 
        should_continue, 
        {
            "end": END, 
            "continue": "reflect_node"}
    )

    # Add edges
    workflow.add_edge("reflect_node", "revise_draft")
        
    return workflow.compile()
    
# Workflow - Long Writing
class State(TypedDict):
    instruction : str
    planning_steps : List[str]
    drafts : List[str]
    final_doc : str
    word_count : int
    revised_drafts: Annotated[list, operator.add]
            
def plan_node(state: State):
    print("###### plan ######")
    instruction = state["instruction"]
    print('subject: ', instruction)
        
    if isKorean(instruction):
        planner_template = (
            "당신은 장문 작성에 능숙한 유능한 글쓰기 도우미입니다."
            "이번 글쓰기는 20,000 단어 이상의 장편을 목표로 합니다."
            "당신은 글쓰기 지시 사항을 여러 개의 하위 작업으로 나눌 것입니다."
            "각 하위 작업은 에세이의 한 단락 작성을 안내할 것이며, 해당 단락의 주요 내용과 단어 수 요구 사항을 포함해야 합니다."

            "글쓰기 지시 사항:"
            "<instruction>"
            "{instruction}"
            "<instruction>"
                
            "다음 형식으로 나누어 주시기 바랍니다. 각 하위 작업은 한 줄을 차지합니다:"
            "1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]"
            "2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]."
            "..."
                
            "각 하위 작업이 명확하고 구체적인지, 그리고 모든 하위 작업이 작문 지시 사항의 전체 내용을 다루고 있는지 확인하세요."
            "과제를 너무 세분화하지 마세요. 각 하위 과제의 문단은 500단어 이상 3000단어 이하여야 합니다."
            "다른 내용은 출력하지 마십시오. 이것은 진행 중인 작업이므로 열린 결론이나 다른 수사학적 표현을 생략하십시오."                
        )
    else:
        planner_template = (
            "You are a helpful assistant highly skilled in long-form writing."
            "This writing aims for a novel of over 20,000 words."
            "You will break down the writing instruction into multiple subtasks."
            "Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph."

            "The writing instruction is as follows:"
            "<instruction>"
            "{instruction}"
            "<instruction>"
                
            "Please break it down in the following format, with each subtask taking up one line:"
            "1. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [Word count requirement, e.g., 800 words]"
            "2. Main Point: [Describe the main point of the paragraph, in detail], Word Count: [word count requirement, e.g. 1500 words]."
            "..."
                
            "Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction."
            "Do not split the subtasks too finely; each subtask's paragraph should be no less than 500 words and no more than 3000 words."
            "Do not output any other content. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks."                
        )
        
    planner_prompt = ChatPromptTemplate([
        ('human', planner_template) 
    ])
                
    chat = get_chat()
        
    planner = planner_prompt | chat
    
    response = planner.invoke({"instruction": instruction})
    print('response: ', response.content)
    
    plan = response.content.strip().replace('\n\n', '\n')
    planning_steps = plan.split('\n')        
    print('planning_steps: ', planning_steps)
            
    return {
        "instruction": instruction,
        "planning_steps": planning_steps
    }
        
def execute_node(state: State):
    print("###### write (execute) ######")        
    instruction = state["instruction"]
    planning_steps = state["planning_steps"]
    print('instruction: ', instruction)
    print('planning_steps: ', planning_steps)
        
    if isKorean(instruction):
        write_template = (
            "당신은 훌륭한 글쓰기 도우미입니다." 
            "아래와 같이 원본 글쓰기 지시사항과 계획한 글쓰기 단계를 제공하겠습니다."
            "또한 제가 이미 작성한 텍스트를 제공합니다."

            "글쓰기 지시사항:"
            "<instruction>"
            "{intructions}"
            "</instruction>"

            "글쓰기 단계:"
            "<plan>"
            "{plan}"
            "</plan>"

            "이미 작성한 텍스트:"
            "<text>"
            "{text}"
            "</text>"

            "글쓰기 지시 사항, 글쓰기 단계, 이미 작성된 텍스트를 참조하여 다음 단계을 계속 작성합니다."
            "다음 단계:"
            "<step>"
            "{STEP}"
            "</step>"
                
            "글이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
            "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
            "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
            "Markdown 포맷으로 서식을 작성하세요."
            "최종 결과에 <result> tag를 붙여주세요."
        )
    else:    
        write_template = (
            "You are an excellent writing assistant." 
            "I will give you an original writing instruction and my planned writing steps."
            "I will also provide you with the text I have already written."
            "Please help me continue writing the next paragraph based on the writing instruction, writing steps, and the already written text."

            "Writing instruction:"
            "<instruction>"
            "{intructions}"
            "</instruction>"

            "Writing steps:"
            "<plan>"
            "{plan}"
            "</plan>"

            "Already written text:"
            "<text>"
            "{text}"
            "</text>"

            "Please integrate the original writing instruction, writing steps, and the already written text, and now continue writing {STEP}."
            "If needed, you can add a small subtitle at the beginning."
            "Remember to only output the paragraph you write, without repeating the already written text."
                
            "Use markdown syntax to format your output:"
            "- Headings: # for main, ## for sections, ### for subsections, etc."
            "- Lists: * or - for bulleted, 1. 2. 3. for numbered"
            "- Do not repeat yourself"
            "Provide the final answer with <result> tag."
        )

    write_prompt = ChatPromptTemplate([
        ('human', write_template)
    ])
        
    text = ""
    drafts = []
    if len(planning_steps) > 50:
        print("plan is too long")
        # print(plan)
        return
        
    for idx, step in enumerate(planning_steps):
        # Invoke the write_chain
        chat = get_chat()
        write_chain = write_prompt | chat
            
        result = write_chain.invoke({
            "intructions": instruction,
            "plan": planning_steps,
            "text": text,
            "STEP": step
        })            
        output = result.content
        # print('output: ', output)
            
        draft = output[output.find('<result>')+8:len(output)-9]
        # print('draft: ', draft) 
                       
        if draft.find('#')!=-1 and draft.find('#')!=0:
            draft = draft[draft.find('#'):]
            
        print(f"--> step:{step}")
        print(f"--> {draft}")
                
        drafts.append(draft)
        text += draft + '\n\n'

    return {
        "instruction": instruction,
        "drafts": drafts
    }

def reflect_draft(conn, reflection_app, idx, draft):     
    inputs = {
        "draft": draft
    }    
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS
    }
    output = reflection_app.invoke(inputs, config)
        
    result = {
        "revised_draft": output['revised_draft'],
        "idx": idx
    }
            
    conn.send(result)    
    conn.close()
        
def reflect_drafts_using_parallel_processing(drafts):
    revised_drafts = drafts
        
    processes = []
    parent_connections = []
        
    reflection_app = buildReflection()
                
    for idx, draft in enumerate(drafts):
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        process = Process(target=reflect_draft, args=(child_conn, reflection_app, idx, draft))
        processes.append(process)
            
    for process in processes:
        process.start()
                
    for parent_conn in parent_connections:
        result = parent_conn.recv()

        if result is not None:
            print('result: ', result)
            revised_drafts[result['idx']] = result['revised_draft']

    for process in processes:
        process.join()
                
    final_doc = ""   
    for revised_draft in revised_drafts:
        final_doc += revised_draft + '\n\n'
        
    return final_doc

def get_subject(query):
    system = (
        "Extract the subject of the question in 6 words or fewer."
    )
        
    human = "<question>{question}</question>"
        
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
        
    chat = get_chat()
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "question": query
            }
        )        
        subject = result.content
        # print('the subject of query: ', subject)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")        
    return subject
    
def markdown_to_html(body):
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <md-block>
    </md-block>
    <script type="module" src="https://md-block.verou.me/md-block.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.css" integrity="sha512-n5zPz6LZB0QV1eraRj4OOxRbsV7a12eAGfFcrJ4bBFxxAwwYDp542z5M0w24tKPEhKk2QzjjIpR5hpOjJtGGoA==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
</head>
<body>
    <div class="markdown-body">
        <md-block>{body}
        </md-block>
    </div>
</body>
</html>"""        
    return html

def revise_answers(state: State):
    print("###### revise_answers ######")
    drafts = state["drafts"]        
    print('drafts: ', drafts)
        
    # reflection
    if multi_region == 'enable':  # parallel processing
        final_doc = reflect_drafts_using_parallel_processing(drafts)
    else:
        reflection_app = buildReflection()
                
        final_doc = ""   
        for idx, draft in enumerate(drafts):
            inputs = {
                "draft": draft
            }    
            config = {
                "recursion_limit": 50,
                "max_revisions": MAX_REVISIONS
            }
            output = reflection_app.invoke(inputs, config)
                
            final_doc += output['revised_draft'] + '\n\n'

    subject = get_subject(state['instruction'])
    subject = subject.replace(" ","_")
    subject = subject.replace("?","")
    subject = subject.replace("!","")
    subject = subject.replace(".","")
    subject = subject.replace(":","")
        
    # markdown file
    markdown_key = 'markdown/'+f"{subject}.md"
    # print('markdown_key: ', markdown_key)
        
    markdown_body = f"## {state['instruction']}\n\n"+final_doc
                
    s3_client = boto3.client('s3')  
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=markdown_key,
        ContentType='text/markdown',
        Body=markdown_body.encode('utf-8')
    )
    # print('response: ', response)
        
    markdown_url = f"{path}{markdown_key}"
    print('markdown_url: ', markdown_url)
        
    # html file
    html_key = 'markdown/'+f"{subject}.html"
        
    html_body = markdown_to_html(markdown_body)
    print('html_body: ', html_body)
        
    s3_client = boto3.client('s3')  
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=html_key,
        ContentType='text/html',
        Body=html_body
    )
    # print('response: ', response)
        
    html_url = f"{path}{html_key}"
    print('html_url: ', html_url)
        
    return {
        "final_doc": final_doc+f"\n<a href={html_url} target=_blank>[미리보기 링크]</a>\n<a href={markdown_url} download=\"{subject}.md\">[다운로드 링크]</a>"
    }
        
def buildLongFormWriting():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("revise_answers", revise_answers)

    # Set entry point
    workflow.set_entry_point("plan_node")

    # Add edges
    workflow.add_edge("plan_node", "execute_node")
    workflow.add_edge("execute_node", "revise_answers")
    workflow.add_edge("revise_answers", END)
        
    return workflow.compile()

def run_long_form_writing_agent(connectionId, requestId, query):    
    app = buildLongFormWriting()
    
    # Run the workflow
    isTyping(connectionId, requestId)        
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 50
    }
    
    output = app.invoke(inputs, config)
    print('output (run_long_form_writing_agent): ', output)
    
    return output['final_doc']

####################### LangGraph #######################
# Long form Writing Agent (Map Reduce Parallel) <-- Not recommended, sometimes it give a empty state by Send() API.
#########################################################

def continue_to_revise(state: State):
    print('###### continue_to_revise ######')
    print('state (continue_to_revise): ', state)
    
    revise_request = []
    for idx, draft in enumerate(state["drafts"]):
        print(f"draft[{idx}]: {draft}")
        
        if draft:
            revise_request.append(Send("revise_node", {
                "draft": draft,
                "idx": idx
            }))
    
    print('revise_request: ', revise_request)
    
    return revise_request

class ReviseState(TypedDict):
    draft: str
    idx: int
    
def revise_node(state: ReviseState):
    if not "idx" or not "draft" in state:
        print("state is None")
        print(state)        
        return 
    
    print(f"###### revise_node {state['idx']} ######")
    print(f"revise_node --> draft[{state['idx']}]: {state['draft']}")    
    
    draft = state["draft"]        
            
    reflection_app = buildReflection()
                    
    inputs = {
        "draft": draft
    }    
    config = {
        "recursion_limit": 50,
        "max_revisions": MAX_REVISIONS
    }
    output = reflection_app.invoke(inputs, config)
    # print('output (revise_node): ', output)
                    
    revised_draft = output['revised_draft']
    print('revised_draft (revise_node): ', revised_draft)
        
    return {
        "revised_drafts": revised_draft
    }
    
def save_answer(state: State):
    print("###### save_answer ######")
    revised_drafts = state["revised_drafts"]        
    print('revised_drafts: ', revised_drafts)
    
    instruction = state['instruction']
    print('instruction: ', instruction)
        
    final_doc = ""
    for idx, revised_draft in enumerate(revised_drafts):
        if revised_draft:
            final_doc += revised_draft + '\n\n'

    subject = get_subject(instruction)
    subject = subject.replace(" ","_")
    subject = subject.replace("?","")
    subject = subject.replace("!","")
    subject = subject.replace(".","")
    subject = subject.replace(":","")
        
    # markdown file
    markdown_key = 'markdown/'+f"{subject}.md"
    # print('markdown_key: ', markdown_key)
        
    markdown_body = f"## {instruction}\n\n"+final_doc
                
    s3_client = boto3.client('s3')  
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=markdown_key,
        ContentType='text/markdown',
        Body=markdown_body.encode('utf-8')
    )
    # print('response: ', response)
        
    markdown_url = f"{path}{markdown_key}"
    print('markdown_url: ', markdown_url)
        
    # html file
    html_key = 'markdown/'+f"{subject}.html"
        
    html_body = markdown_to_html(markdown_body)
    print('html_body: ', html_body)

    s3_client = boto3.client('s3')  
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=html_key,
        ContentType='text/html',
        Body=html_body
    )
    # print('response: ', response)
        
    html_url = f"{path}{html_key}"
    print('html_url: ', html_url)
        
    return {
        "final_doc": final_doc+f"\n<a href={html_url} target=_blank>[미리보기 링크]</a>\n<a href={markdown_url} download=\"{subject}.md\">[다운로드 링크]</a>"
    }    
    
def buildLongFormWritingMapReduce():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("revise_node", revise_node)
    workflow.add_node("save_answer", save_answer)
    
    # Set entry point
    workflow.set_entry_point("plan_node")
        
    workflow.add_conditional_edges(
        "execute_node", 
        continue_to_revise, 
        ["revise_node"]
    )
    
    # Add edges
    workflow.add_edge("plan_node", "execute_node")
    workflow.add_edge("execute_node", "revise_node")
    workflow.add_edge("revise_node", "save_answer")
    workflow.add_edge("save_answer", END)
        
    return workflow.compile()

def run_long_form_writing_agent_map_reduce(connectionId, requestId, query):    
    app = buildLongFormWritingMapReduce()
    
    # Run the workflow
    isTyping(connectionId, requestId)        
    inputs = {
        "instruction": query
    }    
    config = {
        "recursion_limit": 50
    }
    
    output = app.invoke(inputs, config)
    print('output (run_long_form_writing_agent_map_reduce): ', output)
    
    return output['final_doc']
                
#########################################################
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def revise_question(connectionId, requestId, chat, query):    
    global history_length, token_counter_history    
    history_length = token_counter_history = 0
        
    if isKorean(query)==True :      
        system = (
            ""
        )  
        human = """이전 대화를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다. 결과는 <result> tag를 붙여주세요.
        
        <question>            
        {question}
        </question>"""
        
    else: 
        system = (
            ""
        )
        human = """Rephrase the follow up <question> to be a standalone question. Put it in <result> tags.
        <question>            
        {question}
        </question>"""
            
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "history": history,
                "question": query,
            }
        )
        generated_question = result.content
        
        revised_question = generated_question[generated_question.find('<result>')+8:len(generated_question)-9] # remove <result> tag                   
        print('revised_question: ', revised_question)
        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
            
        sendDebugMessage(connectionId, requestId, f"새로운 질문: {revised_question}\n * 대화이력({str(history_length)}자, {token_counter_history} Tokens)을 활용하였습니다.")
            
    return revised_question    
    # return revised_question.replace("\n"," ")

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)
            
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event
            
            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg
    
def sendMessage(id, body):
    # print('sendMessage size: ', len(body))
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        # raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)
    
def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)    
        
def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)    

def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            memory_chain.chat_memory.add_ai_message(msg)     

def translate_text(chat, text):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(chat, text):
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        print('result of grammer correction: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return msg

def use_multimodal(img_base64, query):    
    multimodal = get_multimodal()
    
    if query == "":
        query = "그림에 대해 상세히 설명해줘."
    
    messages = [
        SystemMessage(content="답변은 500자 이내의 한국어로 설명해주세요."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = multimodal.invoke(messages)
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def extract_text(chat, img_base64):    
    query = "텍스트를 추출해서 utf8로 변환하세요. <result> tag를 붙여주세요."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    try: 
        result = chat.invoke(messages)
        
        extracted_text = result.content
        print('result of text extraction from an image: ', extracted_text)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return extracted_text

def getResponse(connectionId, jsonBody):
    print('jsonBody: ', jsonBody)
    
    userId  = jsonBody['user_id']
    print('userId: ', userId)
    requestId  = jsonBody['request_id']
    print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    print('requestTime: ', requestTime)
    type  = jsonBody['type']
    print('type: ', type)
    body = jsonBody['body']
    print('body: ', body)
    convType = jsonBody['convType']
    print('convType: ', convType)
    
    global multi_region    
    if "multi_region" in jsonBody:
        multi_region = jsonBody['multi_region']
    print('multi_region: ', multi_region)
    
    global rag_state    
    if "rag" in jsonBody:
        rag_state = jsonBody['rag']
    else:
        rag_state = 'disable'
    print('rag_state: ', rag_state)
        
    print('initiate....')
    global reference_docs
    reference_docs = []

    global map_chain, memory_chain
    
    # Multi-LLM
    if multi_region == 'enable':
        profile = multi_region_models[selected_chat]
    else:
        profile = LLM_for_chat[selected_chat]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    # print(f'selected_chat: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)
    
    chat = get_chat()    
    
    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)
    
    start = int(time.time())    

    msg = ""
    reference = ""
    
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)    
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:            
                if convType == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)                  

                elif convType == 'long-form-writing-agent':  # long writing
                    msg = run_long_form_writing_agent(connectionId, requestId, text)

                elif convType == 'long-form-writing-agent-map-reduce':  # long writing (map reduce)
                    msg = run_long_form_writing_agent_map_reduce(connectionId, requestId, text)

                elif convType == "translation":
                    msg = translate_text(chat, text) 
                
                elif convType == "grammar":
                    msg = check_grammer(chat, text)  
                                    
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)
                
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            object = body
            file_type = object[object.rfind('.')+1:len(object)]            
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                if not convType == "bedrock-agent":
                    docs = load_csv_document(object)
                    contexts = []
                    for doc in docs:
                        contexts.append(doc.page_content)
                    print('contexts: ', contexts)
                
                    msg = get_summary(chat, contexts)

            elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                # 'page':i+1,
                                'url': path+doc_prefix+parse.quote(object)
                            }
                        )
                    )
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))

                contexts = []
                for doc in docs:
                    contexts.append(doc.page_content)
                print('contexts: ', contexts)

                msg = get_summary(chat, contexts)
                                
            elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
                print('multimodal: ', object)
                
                s3_client = boto3.client('s3') 
                    
                image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+object)
                # print('image_obj: ', image_obj)
                
                image_content = image_obj['Body'].read()
                img = Image.open(BytesIO(image_content))
                
                width, height = img.size 
                print(f"width: {width}, height: {height}, size: {width*height}")
                
                isResized = False
                while(width*height > 5242880):                    
                    width = int(width/2)
                    height = int(height/2)
                    isResized = True
                    print(f"width: {width}, height: {height}, size: {width*height}")
                
                if isResized:
                    img = img.resize((width, height))
                
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                command = ""        
                if 'command' in jsonBody:
                    command  = jsonBody['command']
                    print('command: ', command)
                
                # verify the image
                msg = use_multimodal(img_base64, command)       
                
                # extract text from the image
                text = extract_text(chat, img_base64)
                extracted_text = text[text.find('<result>')+8:len(text)-9] # remove <result> tag
                print('extracted_text: ', extracted_text)
                if len(extracted_text)>10:
                    msg = msg + f"\n\n[추출된 Text]\n{extracted_text}\n"
                
                memory_chain.chat_memory.add_user_message(f"{object}에서 텍스트를 추출하세요.")
                memory_chain.chat_memory.add_ai_message(extracted_text)
            
            else:
                msg = "uploaded file: "+object
        
        sendResultMessage(connectionId, requestId, msg+reference)
        # print('msg+reference: ', msg+reference)    
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg+reference}
        }
        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            # raise Exception ("Not able to write into dynamodb")         
        #print('resp, ', resp)

    return msg, reference

def lambda_handler(event, context):
    # print('event: ', event)
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg, reference = getResponse(connectionId, jsonBody)

                    print('msg+reference: ', msg+reference)
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }
