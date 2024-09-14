# 글쓰기 챗봇 만들기

## Long Term Writing

전체적인 activity diagram은 아래와 같습니다. 여기에서는 plan and excute 패턴을 가지는 agent와 reflection을 수행하는 agent를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. Multi agent 구조로 구성함으로써 복잡한 workflow를 단순하게 구현할 수 있습니다.

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6fe65b1b-a591-4eae-af28-4b5d028774c5">

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
```

이때 "Advanced RAG란?"와 같은 질문에 의해 생성된 Plan의 예는 아래와 같습니다. 

```python
1. Main Point: Advanced RAG(Retrieval Augmented Generation)는 대규모 언어 모델의 지식 증강 기술로, 외부 데이터베이스나 지식원으로부터 관련 정보를 검색하여 언어 모델의 생성 능력을 향상시키는 방법입니다. 이 기술의 핵심 아이디어와 작동 원리, 그리고 기존 언어 모델과의 차이점을 설명합니다. Word Count: 800 words
2. Main Point: Advanced RAG의 구조와 구성 요소(retriever, reader, generator 등)에 대해 자세히 설명하고, 각 구성 요소의 역할과 상호 작용 방식을 설명합니다. 또한 RAG 모델 학습 과정과 관련 기술(retrieval, re-ranking, marginalized augmented generation 등)에 대해 설명합니다. Word Count: 2000 words  
3. Main Point: Advanced RAG의 다양한 응용 분야와 사례(질의 응답, 요약, 데이터 증강 등)를 소개하고, 각 분야에서 RAG가 어떻게 활용되는지 구체적인 예시를 들어 설명합니다. Word Count: 1500 words
4. Main Point: Advanced RAG의 장단점과 한계점, 그리고 향후 발전 방향에 대해 논의합니다. RAG 기술의 윤리적, 사회적 영향과 도전 과제에 대해서도 다룹니다. Word Count: 1200 words
```

Plan 노드에서 생성된 글쓰기 작성 계획을 이용하여 Execute 노드에서는 아래와 같이 각 단락을 작성합니다. 단락 작성시 처음 요청된 글쓰기 지시사항, 전체 글쓰기 단계와 이전 단계에서 작성한 텍스트를 제공한 후에 현재 Step을 주고 이어서 작성하도록 요청합니다. LLM에게 글쓰기 지시사항, 전체 글쓰기 단계, 작성된 텍스트를 제공함으로써 이전에 작성된 글의 문맥을 잊어버리지 않고 원하는 문단을 작성하도록 할 수 있습니다. 이러한 방식은 매 단락 작성시 이전 문장 전체를 입력 context에 제공해야 함으로써 사용되는 token수의 증가와 전체 문장의 내용이 입력 context로 제한됩니다. Anthropic의 Claude3의 경우에 200k token을 제공하므로써 MS Word 기준으로 10페이지 내외의 문서는 작성 가능하지만 이보다 큰 용도로 사용하기 위해서는 이전 글씨나 단계의 내용을 조정하거나 요약하는 방법을 이용합니다. 

```python
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
```


## Revise를 통해 글쓰기 개선

일반적인 사람들의 글쓰기 처럼, 초안 작성후 지속적인 수정을 통해 글의 품질을 향상시킬 수 있습니다. 이러한 글쓰기에서 사람들은 다른 도서나, 인터넷등을 찾아서 참조함으로써 글의 완성도를 높일 수 있습니다. 

revise_node에서는 drafts를 각각 reflect_node에서 reflections을 추출합니다. 또한 이때 최대 3개의 search_queries도 함께 추출하여 검색을 통해 contents를 수집합니다. reflection과 search_queries에 대한 contents를 이용하여 revise_answer에서는 질문을 업데이트합니다. 

![image](https://github.com/user-attachments/assets/be4efa7d-8e93-419e-a46c-2c0eb9f41400)


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

