# 글쓰기 챗봇 만들기

여기에서는 LLM을 이용하여 MS Word 기준으로 약 10페이지 정도의 글쓰기를 가능하게 하는 Agent를 구현하는 방법에 대해 설명합니다.

## Long Term Writing

전체적인 activity diagram은 아래와 같습니다. 여기에서는 plan and excute 패턴을 가지는 agent와 reflection을 수행하는 agent를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. Multi agent 구조로 구성함으로써 복잡한 workflow를 단순하게 구현할 수 있습니다. 이러한 구조는 [essay-writer](https://github.com/kyopark2014/langgraph-agent/blob/main/essay-writer.md#easy-writer)의 multi agent와 유사한 방식으로서, 워크플로우를 2개로 분리하여, 워크플로우별로 최적화가 가능하도록 구조를 개선하였습니다. 또한 Reflection에 대한 워크플로우는 독립되어 실행가능함으로 아래와 같이 Plan and Execute에서 작성된 초안들(dfrafts)을 병렬로 처리할 수 있으므로 동작 속도를 개선할 수 있습니다.

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

우측 Reflection 워크플로우와 같이 각 문단은 Reflection 패턴을 이용하여 문장을 향상시킵니다. 이때 Reflection에 대한 워크플로우는 아래와 같습니다.

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

LLM의 output token과 관련하여 [Anthropic의 Claude3의 경우](https://docs.anthropic.com/en/docs/about-claude/models)의 경우에 4k를 제공하고 있습니다. 일반적인 Q&A에서는 충분한 크기이지만, 장문의 글은 4k보다는 큰 출력을 요구합니다. 또한, 사람은 긴글을 작성하기 먼저 목차를 정하고 상세한 내용을 채워가는 방식을 일반적으로 사용합니다. 이러한 목적을 위해서는 여기에서는 Plan and execute 패턴을 사용하여 먼저 목차를 정하고 각 세부내용을 작성하고자 합니다. 이러한 패턴은 여러번의 LLM 출력을 이용할 수 있으므로 출력 토큰수의 제한에 영향을 받지 않습니다.


사용자의 instruction은 plan_node에서 n개의 plan을 생성합니다. execution_node는 instruction, plans와 현재의 step을 이용하여 draft를 생성합니다. n개의 draft들이 생성됩니다.

<img width="200" alt="image" src="https://github.com/user-attachments/assets/2020f67e-53bd-4d10-995d-d88c952f7f83">

여기에서는 Plan and write를 위하여 [AgentWrite LangGraph](https://github.com/samwit/agent_tutorials/tree/main/agent_write)을 참조하여, 한국어에 맞게 수정하였습니다. Plan에서는 글씨기 지시사항을 아래와 같이 Main point와 Word Count를 이용하여 구성하도록 예제와 함게 요청합니다. 이렇게 함으로써 각 문단의 구분을 한줄로 처리할 수 있고 작성해야 하는 범위를 LLM에게 지시할 수 있게 됩니다.

```text
1. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [Word count requirement, e.g., 800 words]
2. Main Point: [문단의 주요 내용을 자세히 설명하십시오.], Word Count: [word count requirement, e.g. 1500 words]
```

아래는 구현된 Plan 노드를 보여줍니다. 상세한 코드는 [lambda_function.py](./lambda-chat-ws/lambda_function.py)을 참조합니다. 여기서는 LLM의 Role을 정의하고 글씨의 목표를 제시합니다. 사용자가 채팅창에 입력한 글쓰기의 주제는 instruction으로 LLM에 요청됩니다. 

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

Plan 노드에서 생성된 글쓰기 작성 계획을 이용하여 Execute 노드에서는 아래와 같이 각 단락을 작성합니다. 단락 작성시 처음 요청된 글쓰기 지시사항, 전체 글쓰기 단계와 이전 단계에서 작성한 텍스트를 제공한 후에 현재 Step을 주고 이어서 작성하도록 요청합니다. LLM에게 글쓰기 지시사항, 전체 글쓰기 단계, 작성된 텍스트를 제공함으로써 이전에 작성된 글의 문맥을 잊어버리지 않고 원하는 문단을 작성하도록 할 수 있습니다. 이러한 방식은 매 단락 작성시 이전 문장 전체를 입력 context에 제공해야 함으로써 사용되는 token 수의 증가와 전체 문장의 내용이 입력 context로 제한됩니다. Anthropic의 Claude3의 경우에 200k token을 제공하므로써 MS Word 기준으로 10 페이지 내외의 문서는 작성 가능하지만 이보다 큰 용도로 사용하기 위해서는 이전 글씨나 단계의 내용을 조정하거나 요약하는 방법을 이용합니다. 

작성된 문장은 블로그나 인터넷에 쉽게 올릴수 있도로고 markdown 형태를 이용하였습니다. 아래 작성된 초안(draft)과 같이 markdown 형태는 문단이 "###"으로 구분됩니다. 

```text
### Advanced RAG의 응용 분야와 사례
Advanced RAG는 다양한 분야에서 활용될 수 있으며, 특히 질의 응답, 요약, 데이터 증강 등의 분야에서 큰 잠재력을 보이고 있습니다.
**질의 응답(Question Answering)**은 Advanced RAG의 가장 대표적인 응용 분야입니다. RAG 모델은 주어진 질문에 대해 외부 데이터베이스에서 관련 정보를 검색하고, 이를 바탕으로 정확한 답변을 생성할 수 있습니다. 예를 들어, 의학 분야에서 RAG 모델은 환자의 증상과 관련된 의학 문헌을 검색하여 진단과 치료 방법을 제안할 수 있습니다. 또한 법률 분야에서는 관련 법규와 판례를 검색하여 법적 자문을 제공할 수 있습니다.
**요약(Summarization)** 분야에서도 RAG 모델이 활용될 수 있습니다. 긴 문서나 여러 문서에서 핵심 내용을 추출하고 간결하게 요약하는 작업에서 RAG 모델은 외부 지식원을 활용하여 더 정확하고 포괄적인 요약을 생성할 수 있습니다. 예를 들어, 뉴스 기사를 요약할 때 RAG 모델은 관련 배경 지식을 검색하여 중요한 맥락 정보를 포함시킬 수 있습니다.
**데이터 증강(Data Augmentation)** 분야에서도 RAG 모델이 유용하게 활용될 수 있습니다. 기계 학습 모델을 학습시키기 위해서는 대량의 데이터가 필요한데, RAG 모델을 사용하면 기존 데이터에 외부 지식을 추가하여 데이터를 증강시킬 수 있습니다. 예를 들어, 자연어 처리 모델을 학습시킬 때 RAG 모델을 사용하여 기존 데이터에 관련 백과사전 정보를 추가하면 모델의 성능을 향상시킬 수 있습니다.
이 외에도 Advanced RAG는 정보 추출, 지식 그래프 구축, 대화 시스템 등 다양한 분야에서 활용될 수 있습니다. RAG 모델은 외부 지식원을 효과적으로 활용하여 기존 언어 모델의 한계를 극복하고 더 나은 성능을 제공할 수 있습니다.
```

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

일반적인 사람들의 글쓰기 처럼, 초안 작성후 지속적인 수정을 통해 글의 품질을 향상시킬 수 있습니다. 이러한 글쓰기에서 사람들은 다른 도서나, 인터넷등을 찾아서 참조함으로써 글의 완성도를 높일 수 있습니다. 

revise_node에서는 drafts를 각각 reflect_node에서 reflections을 추출합니다. 또한 이때 최대 3개의 search_queries도 함께 추출하여 검색을 통해 contents를 수집합니다. reflection과 search_queries에 대한 contents를 이용하여 revise_answer에서는 질문을 업데이트합니다. 

![image](https://github.com/user-attachments/assets/be4efa7d-8e93-419e-a46c-2c0eb9f41400)


초안(Draft)에 대한 reflection으로 "missing", "advisable", "superfluous"를 구하고, search_queries를 이용해 검색한 결과(content)를 이용하여 문장을 개선합니다. 이때, Reflection, Research 클래스와 [Structured Output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용합니다. 이 방식은 [Reflexion](https://github.com/kyopark2014/langgraph-agent/blob/main/reflexion-agent.md)의 AnswerQuestion/Reflectin을 참조하였습니다. 검색시의 충분한 정보를 획득하기 위하여 검색어가 영/한 번역을 통해 검색을 수행합니다. 

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
    print('draft: ', draft)
    
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

            print('search_queries (mixed): ', search_queries)
            break
        
    revision_number = state["revision_number"] if state.get("revision_number") is not None else 1
    return {
        "reflection": reflection,
        "search_queries": search_queries,
        "revision_number": revision_number + 1
    }
```

Reflect 노드에서 추출된 reflection과 추가 검색으로 얻어진 content 내용을 바탕으로 revise_draft 노드에서는 아래와 같이 초안(draft)에 대한 개선 작업을 수행합니다. 문단의 개선은 draft, reflection, content를 가지고 수정을 진행합니다. 검색은 RAG + 웹검색 또는 웹검색(tavily)를 사용합니다. 

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
            
    content = []             
    global useEnhancedSearch
    useEnhancedSearch = False   
        
    if useEnhancedSearch:
        for q in search_queries:
            response = enhanced_search(q)     
            content.append(response)                   
    else:
        search = TavilySearchResults(max_results=2)
            
        related_docs = []                        
        for q in search_queries:
            response = search.invoke(q)                
            docs = filtered_docs = []
            for r in response:
                if 'content' in r:
                    content = r.get("content")
                    url = r.get("url")
                        
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                'name': 'WWW',
                                'uri': url,
                                'from': 'tavily'
                            },
                        )
                    )                

            if len(docs):
                filtered_docs = grade_documents(q, docs)                
                if len(filtered_docs):
                    related_docs += filtered_docs
            
        for d in related_docs:
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

얻어진 결과가 실제 관련이 있는지를 확인하기 grade_documents로 관련된 문서만을 추출합니다. 

```python
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = grade_documents_using_parallel_processing(question, documents)
    
    return filtered_docs
```

여기서 문서의 관련도 평가에도 병렬처리를 수행하면 속도를 개선합니다. 각각의 관련도 평가는 GradeDocuments 클래스와 structured output을 아래와 같이 이용합니다.

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


## 실행결과

300초 소요.

![image](https://github.com/user-attachments/assets/59f105f1-4777-471f-8416-5826c703f254)

"AWS의 Cloud를 이용해 ERP를 구축하는 방법에 대해 정리해줘."라고 입력합니다.

![image](https://github.com/user-attachments/assets/5ca3a6cd-fced-418d-b9e2-31842a3d6662)




## LLM Agent을 활용해 글쓰는 법

LLM(Large Language Model) Agent을 활용하여 책을 쓰는 과정은 다음과 같습니다:

1. 책의 전반적인 개념, 장르, 대상 독자층을 Agent에게 설명합니다. 이를 통해 LLM이 개요 생성을 위한 맥락을 이해할 수 있습니다.

2. Agent에게 제공한 정보를 바탕으로 책에 포함될 수 있는 잠재적인 주제, 플롯 포인트, 인물 설정 등을 제안하도록 요청합니다.

3. Agent으로 부터 얻은 초기 아이디어를 바탕으로 주요 섹션과 장으로 구성된 상위 수준의 개요를 생성하도록 요청합니다.

4. 개요를 검토하고 확장, 압축 또는 수정이 필요한 부분에 대해 Agent에게 피드백을 제공합니다. 반복적으로 개선된 개요를 요청합니다.

5. 개요의 각 장 또는 섹션에 대해 LLM에게 더 자세한 설명, 플롯 포인트, 심지어 샘플 글까지 생성하여 그 부분을 구체화하도록 요청할 수 있습니다.

6. Agent의 출력물을 시작점과 프레임워크로 활용하되, 스토리, 인물, 작문 스타일 등에 있어서는 저자 본인의 창의적 아이디어를 더해야 합니다. Agent은 브레인스토밍과 개요 작성 도구로 유용하지만, 최종 책은 저자 본인의 목소리와 비전을 반영해야 합니다.

7. Agent의 한계를 인식하고 인간 전문가의 역할을 중요하게 여깁니다. Agent은 창의성과 독창성에 한계가 있으며, 입력 품질에 따라 출력물의 품질이 좌우됩니다. 또한 맥락과 뉘앙스를 이해하는 데 어려움이 있을 수 있습니다. 따라서 Agent을 저자의 창작 과정을 보조하는 도구로 활용하되, 최종 결과물에 대한 통제권은 저자가 가져야 합니다.

