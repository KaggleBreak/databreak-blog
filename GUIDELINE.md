# DataBreak Creator (Blog) 포스팅 작성방식 가이드라인
- 전 세계에 있는 주요 Data Science 커뮤니티 사이트에 있는 데이터(분석, 엔지니어링, 사이언스) 아티클 자료들을 한글화 하여 데이터뽀개기 회원 분들에게 공유하는 컨트리뷰터가 Creator 운영진의 역할입니다. (최소 월 1회)
- github 블로그 통해서 포스팅이 작성되므로 작업은 Github 저장소를 통해 진행됩니다.


## Issue 작성 방법

- 글을 작성하기 전에 github에 있는 Issue를 사전 작성하여 포스터 작성 계획을 올립니다. 
- github에서 Issue를 작성할 때에는 다음과 같은 내용을 꼭 포함합니다.

#### Title & Topic
- 작성할 포스트의 가제와 대략적인 주제, 키워드에 대해 설명해주세요

#### Upload schedule
- 작성 시작 날짜, 업로드 계획 등 대략의 일정을 남겨주세요

#### Reference
- 원본 아티클 사이트 링크 또는 참고문헌 링크를 남겨주세요

- 포스트 제목은 특정 폴더(${HOME/…/src/content)에 yyyy-mm-dd-title.md 형태로 올립니다. 
- 문서는 기본적으로 마크다운을 사용합니다. 다른 포맷을 이용할 경우에는 정상적인 포스팅이 되지 않을 수 있습니다.


### 주요 Data Science 커뮤니티 사이트
1. Kdnuggets (https://www.kdnuggets.com/)
2. Towards Data Science (https://towardsdatascience.com/)
3. Analyticsvidhya (https://www.analyticsvidhya.com/)
4. Medium (https://medium.com)
5. Kaggle Blog (http://blog.kaggle.com/)

- 원본으로 포스팅 하는 자료는 위에 있는 주요 Data Science 커뮤니티 사이트가 아니여도 괜찮습니다. 
- 논문의 경우에는 arVix(아카이브)에 있는 논문 또는 특정 저널을 참고하시면 됩니다.
- 해당 문서와 관련해 추가로 내용을 참고한 곳이 있다면 **반드시 참고 링크 혹은 출처를 명시해주세요.**

- 이미지 파일은 ${HOME}/src/content/img/{본인 이름 폴더} 권장 디렉터리에 모아둡니다. 
- 이미지 파일은 header 저장하고, 그외는 구글드라이브에 이미지를 업로드 하여 url만 포스트에 저장합니다. 
- header에 들어가는 이미지 파일은 yyyy-mm-dd-title.gif(png...)로 통일합니다.


## 포스팅 작업 순서
1. 새로운 Issue를 발급하며, Title & Topic / Upload schedule / Reference를 적습니다.
2. 자기 자신을 self-assign합니다. 혹은 자신이 assign 되어있는 이슈를 찾습니다.
3. 포스터 문서 작업을 마치면 해당 파일을 push 합니다.
4. push가 완료되면 포스터 리뷰를 합니다. 포스터 리뷰는 포스터 올린 사람이 참여하는 운영진 중을 입력하여 랜덤을 추출하여 지정합니다. [랜덤 추첨기](https://prevl.org/service/dist/random-picker/)
  만약 온라인 리뷰가 안되더라도, 정기적인 Creator 오프라인 모임에서 발표를 진행하면서 오프라인 리뷰를 받습니다. (발표 시간 : 20~30분)
5. 온라인, 오프라인 리뷰가 끝난 포스팅은 페이스북에 직접 올립니다. 페이스북에 포스팅을 올리는 방식은 아래와 같습니다.



## 페이스북에 포스팅을 올리는 방식

1.포스팅 요약할 수 있는 태그 사용 (#, Ex. #GAN, #TSNE)

2.오늘은 … 아티클을 소개합니다.

3.추천 대상을 요약해서 작성하기 (어떤 사람이 해당 아티클을 읽어야 도움이 되는지 명시하기)

4.해당 아티클을 읽었을 때 후기 또는 인사이트 작성하기 (느낀점, 솔직하게, 나의 생각)

5.링크 :

6.자세한건 데이터뽀개기 소모임 000 블로그를 참고하세요!
(데이터뽀개기 포스팅 방법 참고)[https://www.facebook.com/groups/databreak/permalink/2341916956062973/]

- 주요 내용은 제외합니다. 


## Push 방법

* Push는 **각각의 작업단위(Issue 단위)** 의 commit으로 보냅니다.
* Commit message는 `내용 (#이슈넘버)`으로 작성합니다 (한글로 작성해도 무방합니다). ex) 아티클 초안 작성 (#1)
* 참고 : [Git Style Guide](https://github.com/ikaruce/git-style-guide)


## 문서 작성
> 샘플 문서인 [](.md 파일)를 참고해주세요.

*모든 문서는 마크다운(\*.md)으로 작성합니다.
* 중요한 부분은 **\*\*bold\*\***, *\*italiic\** 등을 적절히 활용해 강조합니다.내용의 주제가 나뉘는 경우 대제목과 소제목으로 나눠 가독성을 높힙니다.
* 코드는 코드 블럭으로 묶어서 나타냅니다. 스크린 샷이 필요한 경우 글자가 깨지지 않도록 큰 사이즈로 캡쳐합니다.
* 참고 링크는 하단에 모아둡니다.
* 모두가 함께 보는 글이 될테니 PR 전에 [맞춤법 교정](http://speller.cs.pusan.ac.kr) 하는 것을 추천합니다.


## 추가 사항
* 포스팅 작업을 하다가 어려운 사항이 생기면 망설이지 말고 언제든 이슈 및 슬랙으로 이야기합니다.
