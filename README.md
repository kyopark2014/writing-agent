# Writing Agent 구현하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Fwriting-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>

다음의 내용은 아래와 같은 용도로 LLM Agent를 활용하고자 하는 사람들을 위해 만들었습니다.

1) LLM이 한번에 사용하는 토큰의 숫자가 많더라도 결과가 좋았으면 좋겠다는 사람
2) LLM이 알아서 인터넷 또는 RAG을 검색해서, 충분한 길이를 가지는 글을 작성해주면, blog나 github에 수정없이 바로 올리고 싶은 사람
3) 결과만 좋다면 5분, 10분 정도는 기다려줄 수 있는 사람

여기서 구현하는 LLM Agent는 사람의 글쓰기 사고 과정을 모방함으로써 한번에 MS Word 기준으로 약 10페이지 정도의 글쓰기가 가능합니다. 인터넷과 RAG를 통해 얻어진 결과를 활용하고 markdown 형태로 작성되어 blog나 github에 바로 올릴 수 있으며, html URL 형태로 작성된 문서를 공유할 수 있습니다.

## 구현된 Architecture

LangGraph로 Long term writing을 구현하기 위하여 아래와 같은 serverless architecture를 이용합니다. 이를 통해 트래픽이 없는 경우에는 비용이 거의 발생하지 않으며, 트래픽이 높아질때에는 자동으로 스케일 아웃(Scale out)함으로써 변화하는 트래픽에 효과적으로 대응할 수 있습니다.

1) 사용자의 질문과 답변의 양방향 대화를 원할히 수행할 수 있도록 클라이언트와 애틀리케이션 서버간 연결에 WebSocket을 이용합니다. 이때 WebSocket의 Endpoint는 [WebSocket을 지원하는 API Gateway](https://docs.aws.amazon.com/ko_kr/apigateway/latest/developerguide/apigateway-websocket-api-overview.html)를 이용합니다. 이때 사용자의 입력은 json형태로 message-id와 converstion-type과 같은 정보를 포함합니다. 
2) AWS Lambda는 사용자의 입력을 받으면 LangGraph의 Workflow를 이용하여 순차적으로 명령을 수행합니다. 여기서 사용자의 입력은 글쓰기를 위한 지시사항으로써, LangGraph의 [plan-and-execute 패턴](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/#create-the-graph)에 따라 먼저 단계별로 글쓰는 주제를 선정합니다. 선정된 주제에 따라 초안(draft)를 생성합니다.
3) [초안에 대한 reflection](https://blog.langchain.dev/reflection-agents/)을 통해 초안을 개선하기 위한 포인트를 찾고 검색을 통해 내용을 보강합니다. 여기서는 [Amazon Bedrock의 완전관리형 RAG 서비스인 Knowledge base](https://aws.amazon.com/ko/blogs/korea/knowledge-bases-now-delivers-fully-managed-rag-experience-in-amazon-bedrock/)를 이용하여 RAG를 검색하고 Tavily를 통해 검색한 컨텐츠를 사용합니다. Knowledge base는 Amazon S3에 업로드된 DOC, PDF, PPT뿐 아니라 웹크롤러 데이터 소스를 이용하여 인터넷의 다양한 데이터를 RAG로 사용할 수 있도록 지원합니다.
4) RAG와 웹검색을 통해 얻어진 관련된 문서들이 실제 관련이 있는지를 prompt와 structured output으로 확인하고 관련된 문서에서 컨텐츠를 추출하여 초안(Draft)를 개선합니다. Plan 단계에서 여러개의 Draft들이 생성되었으므로 multi-region LLM을 이용하여 병렬로 속도를 향상시킵니다. 

![image](https://github.com/user-attachments/assets/f2fa332d-0e44-4f92-90e3-0d5d4b5babe5)


## Long Term Writing

전체적인 activity diagram은 아래와 같습니다. 여기에서는 [plan and excute 패턴을 가지는 agent](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)와 [reflection을 수행하는 agent](https://blog.langchain.dev/reflection-agents/)를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. 여기서는 multi agent 구조를 이용하여 복잡한 workflow를 손쉽게 구현합니다. 이러한 구조는 [essay-writer](https://github.com/kyopark2014/langgraph-agent/blob/main/essay-writer.md#easy-writer)의 multi agent와 유사한 방식이지만, 워크플로우를 2개로 분리하여, 워크플로우별로 최적화가 가능하도록 구조를 개선하였고, reflection에 대한 워크플로우를 초안들(dfrafts)의 숫자만큼 병렬로 처리할 수 있으므로 전체적인 동작 속도를 개선할 수 있습니다.

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6fe65b1b-a591-4eae-af28-4b5d028774c5">

여기서 좌측의 Plan and Execute 워크플로우는 아래와 같이 정의합니다.

```python
class State(TypedDict):
    instruction : str
    planning_steps : List[str]
    drafts : List[str]
    final_doc : str
    word_count : int

def buildLongTermWriting():
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("planning_node", plan_node)
    workflow.add_node("execute_node", execute_node)
    workflow.add_node("revising_node", revise_answer)

    # Set entry point
    workflow.set_entry_point("planning_node")

    # Add edges
    workflow.add_edge("planning_node", "execute_node")
    workflow.add_edge("execute_node", "revising_node")
    workflow.add_edge("revising_node", END)
        
    return workflow.compile()
```

우측 reflection 워크플로우와 같이 각 문단은 reflection 패턴을 이용하여 문장을 향상시킵니다. 이때 reflection에 대한 워크플로우는 아래와 같습니다.

```python
class ReflectionState(TypedDict):
    draft : str
    reflection : List[str]
    search_queries : List[str]
    revised_draft: str
    revision_number: int

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
```

## Plan and Execute

LLM의 output token과 관련하여 [Anthropic의 Claude3의 경우](https://docs.anthropic.com/en/docs/about-claude/models)의 경우에 4k를 제공하고 있습니다. 일반적인 질문과 답변(Q&A) 형태에서는 충분한 크기이지만, 장문의 글은 4k보다는 큰 출력을 요구합니다. 또한, 사람들은 긴글을 작성하기위해 목차를 정하고 상세한 내용을 채워가는 방식을 사용합니다. 이러한 사람의 사고방식을 따라서, 여기에서는 plan and execute 패턴을 사용하여 먼저 목차를 정하고 각 세부 내용을 작성합니다. 이러한 패턴은 여러번의 LLM 출력을 이용할 수 있도록 해주므로, LLM 출력 토큰수의 제한을 극복할 수 있습니다.

사용자의 지시사항(instruction)을 이용하여 plan node는 n개의 계획(plan)을 생성합니다. Execution node는 instruction, plans과 현재 step을 이용하여 초안(draft)들을 생성합니다. 

<img width="200" alt="image" src="https://github.com/user-attachments/assets/2020f67e-53bd-4d10-995d-d88c952f7f83">

여기에서는 plan and write를 위하여 [AgentWrite LangGraph](https://github.com/samwit/agent_tutorials/tree/main/agent_write)을 참조하여, 한국어에 맞게 수정하였습니다. Plan에서는 글씨기 지시사항을 main point와 word count를 이용하여 아래와 같이 구성합니다. 아래와 같은 구체적인 예를 제시함으로써, 각 문단의 계획(plan)을 한줄로 정의할 수 있고 작성해야 하는 범위를 LLM에게 구체적으로 지시할 수 있습니다. 

```text
1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]
2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]
```

아래는 구현된 plan 노드를 보여줍니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다. 여기서는 LLM의 role을 정의하고 글쓰기의 목표를 제시합니다. 

```python
def plan_node(state: State):
    print("###### plan ######")
    instruction = state["instruction"]
        
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
    
    plan = response.content.strip().replace('\n\n', '\n')
    planning_steps = plan.split('\n')        
            
    return {
        "instruction": instruction,
        "planning_steps": planning_steps
    }
```

이때 "Advanced RAG란?"와 같은 질문에 의해 생성된 Plan의 예는 아래와 같습니다. 

```python
1. Main Point: Advanced RAG(Retrieval Augmented Generation)는 대규모 언어 모델의 지식 증강 기술로, 외부 데이터베이스나 지식원으로부터 관련 정보를 검색하여 언어 모델의 생성 능력을 향상시키는 방법입니다. 이 기술의 핵심 아이디어와 작동 원리, 그리고 기존 언어 모델과의 차이점을 설명합니다. Word Count: 800 words
2. Main Point: Advanced RAG의 구조와 구성 요소(retriever, reader, generator 등)에 대해 자세히 설명하고, 각 구성 요소의 역할과 상호 작용 방식을 설명합니다. 또한 RAG 모델 학습 과정과 관련 기술(retrieval, re-ranking, marginalized augmented generation 등)에 대해 설명합니다. Word Count: 2000 words  
3. Main Point: Advanced RAG의 다양한 응용 분야와 사례(질의 응답, 요약, 데이터 증강 등)를 소개하고, 각 분야에서 RAG가 어떻게 활용되는지 구체적인 예시를 들어 설명합니다. Word Count: 1500 words
4. Main Point: Advanced RAG의 장단점과 한계점, 그리고 향후 발전 방향에 대해 논의합니다. RAG 기술의 윤리적, 사회적 영향과 도전 과제에 대해서도 다룹니다. Word Count: 1200 words
```

Plan 노드에서 생성된 글쓰기 작성 계획을 이용하여 execute 노드에서는 아래와 같이 각 단락을 작성합니다. 단락 작성시 처음 요청된 글쓰기 지시사항, 전체 글쓰기 단계와 이전 단계에서 작성한 텍스트를 제공한 후에 현재 step을 주고 이어서 작성하도록 요청합니다. LLM에게 글쓰기 지시사항, 전체 글쓰기 단계, 작성된 텍스트를 제공함으로써 이전에 작성된 글의 문맥을 잊어버리지 않고 원하는 문단을 작성하도록 할 수 있습니다. 이러한 방식은 매 단락 작성시 이전 문장 전체를 입력 context에 제공해야 함으로써 사용되는 token 수의 증가와 전체 문장의 내용이 입력 context로 제한됩니다. Anthropic의 Claude 3의 경우에 200k token을 제공하므로 MS Word 기준으로 10 페이지 내외의 문서는 작성 가능하지만 이보다 큰 용도로 사용하기 위해서는 이전 글씨나 단계의 내용을 조정하거나 요약하는 방법을 이용합니다. 

```python
def execute_node(state: State):
    print("###### write (execute) ######")        
    instruction = state["instruction"]
    planning_steps = state["planning_steps"]
        
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
    for idx, step in enumerate(planning_steps):
        chat = get_chat()
        write_chain = write_prompt | chat
            
        result = write_chain.invoke({
            "intructions": instruction,
            "plan": planning_steps,
            "text": text,
            "STEP": step
        })            
        output = result.content
            
        draft = output[output.find('<result>')+8:len(output)-9]
            
        print(f"--> step:{step}")
        print(f"--> {draft}")
                
        drafts.append(draft)
        text += draft + '\n\n'

    return {
        "instruction": instruction,
        "drafts": drafts
    }
```

Execute 노드에서 작성된 각 단락은 초안(draft)이므로 여기에서는 reflection 패턴을 통해 작성된 문단을 향상시킵니다. 작성된 글은 html로 변환하여 Amazon S3에 저장후, 직접 markdown형태로 공유하거나, 별도로 다운로드하여 블로그나 github을 통해 공유 될 수 있습니다. 

```python
def revise_answer(state: State):
    print("###### revise ######")
    drafts = state["drafts"]        
        
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
                "max_revisions": 1
            }
            output = reflection_app.invoke(inputs, config)
                
            final_doc += output['revised_draft'] + '\n\n'

    subject = get_subject(state['instruction'])
    markdown_key = 'markdown/'+f"{subject}.md"
        
    markdown_body = f"## {state['instruction']}\n\n"+final_doc
                
    s3_client = boto3.client('s3')  
    response = s3_client.put_object(
        Bucket=s3_bucket,
        Key=markdown_key,
        ContentType='text/markdown',
        Body=markdown_body.encode('utf-8')
    )
        
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
        
    html_url = f"{path}{html_key}"
    print('html_url: ', html_url)
        
    return {
        "final_doc": final_doc+f"\n<a href={html_url} target=_blank>[미리보기 링크]</a>\n<a href={markdown_url} download=\"{subject}.md\">[다운로드 링크]</a>"
    }
```

작성된 문장은 블로그나 인터넷에 쉽게 올릴수 있도로고 markdown 형태를 이용하였습니다. 아래 작성된 초안(draft)과 같이 markdown 형태는 문단이 "###"으로 구분됩니다. 

```text
### Advanced RAG의 응용 분야와 사례
Advanced RAG는 다양한 분야에서 활용될 수 있으며, 특히 질의 응답, 요약, 데이터 증강 등의 분야에서 큰 잠재력을 보이고 있습니다.
**질의 응답(Question Answering)**은 Advanced RAG의 가장 대표적인 응용 분야입니다. RAG 모델은 주어진 질문에 대해 외부 데이터베이스에서 관련 정보를 검색하고, 이를 바탕으로 정확한 답변을 생성할 수 있습니다. 예를 들어, 의학 분야에서 RAG 모델은 환자의 증상과 관련된 의학 문헌을 검색하여 진단과 치료 방법을 제안할 수 있습니다. 또한 법률 분야에서는 관련 법규와 판례를 검색하여 법적 자문을 제공할 수 있습니다.
**요약(Summarization)** 분야에서도 RAG 모델이 활용될 수 있습니다. 긴 문서나 여러 문서에서 핵심 내용을 추출하고 간결하게 요약하는 작업에서 RAG 모델은 외부 지식원을 활용하여 더 정확하고 포괄적인 요약을 생성할 수 있습니다. 예를 들어, 뉴스 기사를 요약할 때 RAG 모델은 관련 배경 지식을 검색하여 중요한 맥락 정보를 포함시킬 수 있습니다.
**데이터 증강(Data Augmentation)** 분야에서도 RAG 모델이 유용하게 활용될 수 있습니다. 기계 학습 모델을 학습시키기 위해서는 대량의 데이터가 필요한데, RAG 모델을 사용하면 기존 데이터에 외부 지식을 추가하여 데이터를 증강시킬 수 있습니다. 예를 들어, 자연어 처리 모델을 학습시킬 때 RAG 모델을 사용하여 기존 데이터에 관련 백과사전 정보를 추가하면 모델의 성능을 향상시킬 수 있습니다.
이 외에도 Advanced RAG는 정보 추출, 지식 그래프 구축, 대화 시스템 등 다양한 분야에서 활용될 수 있습니다. RAG 모델은 외부 지식원을 효과적으로 활용하여 기존 언어 모델의 한계를 극복하고 더 나은 성능을 제공할 수 있습니다.
```

Reflection 과정은 문장의 개선점을 찾고 부족한 부분은 검색하고 이를 적용하는 작업을 반복함으로써 전체 문단들을 순차적으로 진행하기 보다는 reflect_drafts_using_parallel_processing()와 같이 병렬 처리하는 것이 합리적입니다. 각 문단은 reflect_draft()을 통해 수정 작업을 수행합니다. 문단을 병렬 처리할 경우에 각 문단의 개선 작업 시간은 각 문단의 길이와 검색하는 컨텐츠의 양에 따라 달라집니다. 따라서, 아래와 같이 문단에 대한 개선작업 요청에 문서의 인덱스를 포함하고, 결과를 json 형태로 받아서 각 index에 따라 수정된 문단(revised_draft)을 배치합니다. 

```python
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
            revised_drafts[result['idx']] = result['revised_draft']

    for process in processes:
        process.join()
                
    final_doc = ""   
    for revised_draft in revised_drafts:
        final_doc += revised_draft + '\n\n'
        
    return final_doc
```

작성된 markdown 문서를 html로 제공하기 위해서 아래와 같이 <md-block> 태그를 이용합니다. 또한 글쓰기 패턴은 github 형태를 이용하였습니다. 

```python
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
```

## Revise를 통해 글쓰기 개선

일반적인 사람들의 글쓰기처럼, 초안 작성후 지속적인 수정을 통해 글의 품질을 향상시킬 수 있습니다. 이러한 글쓰기에서 사람들은 다른 도서나, 인터넷등을 찾아서 참조함으로써 글의 완성도를 높일 수 있습니다. 

Revise node에서는 drafts를 각각 reflect node에서 reflections을 추출합니다. 또한 이때 최대 3개의 search_queries도 함께 추출하여 검색을 통해 contents를 수집합니다. reflection과 search_queries에 대한 contents를 이용하여 revise_answer에서는 질문을 업데이트합니다. 

![image](https://github.com/user-attachments/assets/be4efa7d-8e93-419e-a46c-2c0eb9f41400)


초안(Draft)에 대한 reflection으로 "missing", "advisable", "superfluous"를 구하고, search_queries를 이용해 검색한 결과(content)를 이용하여 문장을 개선합니다. 이때, reflection, research 클래스와 [Structured Output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용합니다. 이 방식은 [Reflexion](https://github.com/kyopark2014/langgraph-agent/blob/main/reflexion-agent.md)의 AnswerQuestion/Reflectin을 참조하였습니다. 검색시의 충분한 정보를 획득하기 위하여 검색어가 영/한 번역을 통해 검색을 수행합니다. 

```python
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
    
def reflect_node(state: ReflectionState):
    print("###### reflect ######")
    draft = state['draft']
    
    reflection = []
    search_queries = []
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Research, include_raw=True)
            
        info = structured_llm.invoke(draft)                
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            search_queries = parsed_info.search_queries
                
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
            break
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "reflection": reflection,
        "search_queries": search_queries,
        "revision_number": revision_number + 1
    }
```

Reflect 노드에서 추출된 reflection과 추가 검색으로 얻어진 content 내용을 바탕으로 revise_draft 노드에서는 아래와 같이 초안(draft)에 대한 개선 작업을 수행합니다. 문단의 개선은 draft, reflection, content를 가지고 수정을 진행합니다. 검색은 RAG와 웹 검색(tavily)를 사용합니다. 

```python
def revise_draft(state: ReflectionState):   
    print("###### revise_answer ######")
        
    draft = state['draft']
    search_queries = state['search_queries']
    reflection = state['reflection']
        
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
    for q in search_queries:
        docs = retrieve_from_knowledge_base(q)
        print(f'q: {q}, RAG: {docs}')
        
        if len(docs):
            filtered_docs += grade_documents(q, docs)
    
    # web search
    search = TavilySearchResults(max_results=2)
    for q in search_queries:
        response = search.invoke(q)
                
        docs = []
        for r in response:
            if 'content' in r:                        
                docs.append(
                    Document(
                        page_content=r.get("content"),
                        metadata={
                            'name': 'WWW',
                            'uri': r.get("url"),
                            'from': 'tavily'
                        },
                    )
                )                
        if len(docs):
            filtered_docs += grade_documents(q, docs)
        
    content = []   
    if len(filtered_docs):
        for d in filtered_docs:
            content.append(d.page_content)
        
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
        
    revised_draft = output[output.find('<result>')+8:len(output)-9]
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
        
    return {
        "revised_draft": revised_draft,
        "revision_number": revision_number
    }
```

얻어진 결과가 실제 관련이 있는지를 확인하기 grade_documents()로 관련된 문서만을 추출합니다. 

```python
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = grade_documents_using_parallel_processing(question, documents)
    
    return filtered_docs
```

여기서 문서의 관련도 평가에도 병렬처리를 수행하면 속도를 개선합니다. 각각의 관련도 평가는 gradeDocuments 클래스와 structured output을 아래와 같이 이용합니다.

```python
def grade_documents_using_parallel_processing(question, documents):
    global selected_chat
    
    filtered_docs = []    

    processes = []
    parent_connections = []
    
    for i, doc in enumerate(documents):
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

def grade_document_based_on_relevance(conn, question, doc, models, selected):     
    chat = get_multi_region_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    
    grade = score.binary_score    
    if grade == 'yes':
        print("---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)
    
    conn.close()
```

문단에 대한 평가 및 개선은 MAX_REVISIONS 만큰 수행합니다. 아래와 같이 should_continue()은 conditional edge로서 반복 숫자에 대한 조건문을 포함하고 있습니다. 

```python
MAX_REVISIONS = 1
def should_continue(state: ReflectionState, config):
    print("###### should_continue ######")
    max_revisions = config.get("configurable", {}).get("max_revisions", MAX_REVISIONS)
    print("max_revisions: ", max_revisions)
            
    if state["revision_number"] > max_revisions:
        return "end"
    return "continue"
```

### RAG 사용

여기에서는 Amazon Bedrock의 완전관리형 RAG 서비스인 Knowledge Base를 이용하고 있습니다. Knowledge Base는 Amazon S3에 파일을 올리거나, web crawler를 이용해 하위 URL까지 문서를 가져올 수 있어서 편리합니다. 설치하는 방법은 [Knowledge Base로 Advanced RAG 구현](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/knowledge-base.md)을 참조합니다. 

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)



### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


## 실행결과

"VPC와 VPC를 안전하게 연결하고 연결후 동작을 확인하는 방법"이라고 입력하면 아래와 같은 결과를 얻을 수 있습니다.

![image](https://github.com/user-attachments/assets/4ac97942-4db8-4cb5-bd06-d0a4a6a4633e)

하단의 [미리보기 링크]을 선택하면 [markdown 파일](./contents/VPC_%EA%B0%84_%EC%95%88%EC%A0%84%ED%95%9C_%EC%97%B0%EA%B2%B0_%EB%B0%8F_%ED%99%95%EC%9D%B8.md)의 내용을 아래와 같이 확인할 수 있습니다.

![image](https://github.com/user-attachments/assets/14fbb094-21ba-4d92-931f-423d5f6f4872)

이때의 동작은 아래와 같이 LangSmith에서 확인할 수 있습니다. 여기에서는 결과를 얻기까지 약140초가 소요되었습니다. 

![image](https://github.com/user-attachments/assets/566981f1-ba1b-4219-a6d1-5e36595d4809)

### 결과 예제

- [AWS의 생성형 AI 서비스에서 데이터를 수집하는 방법](./contents/AWS_AI_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%88%98%EC%A7%91_%EB%B0%A9%EB%B2%95.md)

- [AWS Security Hub, Amazon GuardDuty와 Azure Sentinel을 비교해주세요. AWS 서비스가 Azure Sentinel 대비 강점도 자세히 알려주세요.](./contents/Cloud_security_monitoring_and_threat_detection.md)

- [adipocyte cells (3T3L1)과 macrophages co-culutre 실험을 어떻게 design할수 있을까 ?](./contents/Co-culturing_adipocytes_and_macrophages.md)

- [지방 조직이 분비하는 exosome들이 어떻게 면역체계에 역할을 하고 어떻게 하면 좋은 exosome들을 분비시켜 당뇨나 병을 예방할수 있는지 알려주세요.](./contents/Exosomes_from_fat_tissue_regulate_immunity.md)

- [AWS에서 ERP를 Cloud로 구축하는 방법](./contents/Deploying_ERP_on_AWS_Cloud.md)

- [우주 여행을하면서 우주인을 만나는 여행에 대한 얘기를 해주세요. 여행지는 목성, 금성, 토성 입니다. 각 여행지에는 특별한 형태와 성격이 다양한 외계인이 있어요. 이들은 성경이 좋기도하고 괴팍하기도 하고 항상 슬프거나 즐겁기도 합니다. 각 별의 외계인의 성격은 마음껏 상상해도 됩니다. 우리는 이들과 우정을 쌓으면서 여행을 하게 되고 마지막에는 지구로 함께 돌아와 재미있게 놀 예정이에요.](./contents/Meeting_aliens_on_space_travel..md)

- [Python으로 생성한 텍스트 파일을 열었을때 한글이 깨지는 경우에 대한 대응방법](./contents/Text_file_encoding_issue_in_Python.md)

- [판타지 소설을 써줘. 간달프의 젋은 시절이 배경으로 그의 로멘스가 중심이 되었으면 좋겠어. 그는 뉴욕에 살명서 한국여자를 사랑하게 돼. 그래서 서울에 와서 즐거운 모험을 하는데 갑자기 제주에 팬션을 열었어. 거기서 이효리랑 친구가 되어서 나중에는 보이그룹으로 데뷰를 하고 이후에 일본에 가서 본격적인 활동을하는 내용이야.](./contents/%EA%B0%84%EB%8B%AC%ED%94%84%EC%9D%98_%EB%A1%9C%EB%A7%A8%ED%8B%B1_%EB%AA%A8%ED%97%98%EA%B3%BC_%EC%95%84%EC%9D%B4%EB%8F%8C_%ED%99%9C%EB%8F%99.md)

- [Bedrock agent에 대해 설명해주세요.](https://github.com/kyopark2014/writing-agent/blob/main/contents/%EB%B2%A0%EB%93%9C%EB%A1%9D_%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8%EB%8A%94_%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5_%EA%B8%B0%EB%B0%98_AI_%EC%8B%9C%EC%8A%A4%ED%85%9C%EC%9E%85%EB%8B%88%EB%8B%A4.md)

- [VPC와 VPC를 안전하게 연결하는 방법과, 연결이 잘되었는지 확인하는 방법에 대해 설명해주세요.](./contents/VPC_%EA%B0%84_%EC%95%88%EC%A0%84%ED%95%9C_%EC%97%B0%EA%B2%B0_%EB%B0%8F_%ED%99%95%EC%9D%B8%EB%B0%A9%EB%B2%95.md)

  


