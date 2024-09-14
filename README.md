# 글쓰기 챗봇 만들기

## Long Term Writing

전체적인 activity diagram은 아래와 같습니다. 여기에서는 plan and excute 패턴을 가지는 agent와 reflection을 수행하는 agent를 이용하여 instruction으로 장문의 글쓰기를 수행합니다. Multi agent 구조로 구성함으로써 복잡한 workflow를 단순하게 구현할 수 있습니다.

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6fe65b1b-a591-4eae-af28-4b5d028774c5">

사용자의 instruction은 plan_node에서 n개의 plan을 생성합니다. execution_node는 instruction, plans와 현재의 step을 이용하여 draft를 생성합니다. n개의 draft들이 생성됩니다.

<img width="200" alt="image" src="https://github.com/user-attachments/assets/2020f67e-53bd-4d10-995d-d88c952f7f83">

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

