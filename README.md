# BERT-Additional-Classification-Head


# Dataset 

BERTSUM 을 이용하여 Extarctive Sumaarization 추출 ( Ratio = 0.5 ) 


# MODEL  


본 연구에서는 원문에서 생성된 추출 요약문 을 통하여 언어 모델 감성 분류 성능을 향상시키는 방법을 제안

감성분류에 있어 기존 문서에서 감성 분석을 진행 할 때 불필요한 정보를 제거하고 감성 분류를 진행하면 감성
분류의 성능이 높아짐을 기대


따라서 요약 정보를 이용하여 분류모델에 사용하는 방법을 제안

![image](https://user-images.githubusercontent.com/76906638/168416938-92cd54f1-7594-490b-a20b-1c524b3c04cd.png)


# FrameWork 

![image](https://user-images.githubusercontent.com/76906638/168416948-6dfc3b78-d287-4be0-83b1-f5ddb58ef0aa.png)

추출 된 요약문을 어텐션 마스크 와 같은 방식으로 원 문서에 중요 문장으로 식별한 문장 들을 1로 마스킹 하며
이외의 문장들은 0으로 패딩 처리 진행

원본 문서 에서 추출된 요약문과 원본 문서 데이터를 인풋으로 사용하여 Fine Tuning 진행

![image](https://user-images.githubusercontent.com/76906638/168416963-b3fb54b6-3809-4f4b-b752-899f91cdc97f.png)



#  Dataset 

Base Model 과 Summary include Model 은 사전 학습 된 BERT – Base Model 사용

BERT - Base Model
 • Encoder Block : 12


Data set : IMDB Dataset ( 0 : Negative , 1 : Positive )
 • Train Data set : 19872
 • Validation Data : 4969
 • Test Data : 25000


# Conclusion 

Summary 를 포함했을 때 Base line 보다 성능이 감소하는 것을 확인

Summary 는 일반화 성능에 부정적인 영향을 미침



![image](https://user-images.githubusercontent.com/76906638/168417010-1ac4dbb2-f6fd-43b1-9b8e-9bbea8178639.png)

