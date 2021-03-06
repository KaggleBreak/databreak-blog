---
layout: post
title: "It’s Only Natural: An Excessively Deep Dive Into Natural Gradient Optimization(KR)"
image: img/youngpyoryu/Image Credit.jpeg
author: youngpyoryu
date: "2019-04-14T15:46:37.121Z"
tags: 
  - Optimization
---


# It’s Only Natural: An Excessively Deep Dive Into Natural Gradient Optimization

## 참고자료 
아래의 글은 Cody Marie Wild 의 [It’s Only Natural: An Excessively Deep Dive Into Natural Gradient Optimization] (https://towardsdatascience.com/its-only-natural-an-excessively-deep-dive-into-natural-gradient-optimization-75d464b89dbb)를 번역한 글입니다.



1차 근사를 위한 모든 현대 심층 학습 모델은 경사하강법을 이용하여 훈련됩니다. 경사하강법의 각 단계에서 매개변수 값들은 시작 지점에서 시작되고 손실을 가장 많이 감소시키는 방향으로 그 값들을 이동시킵니다. 당신은 매개 변수들의 전체 벡터로 당신의 손실을 미분함으로써 이를 수행합니다, 이를 다른 말로 Jacobian이라고 합니다. 
그러나 이것은 손실의 1계 미분일 뿐이며 곡률이나 당신의 1차 미분이 얼마나 빨리 변하는 지에 대해선 알려주지 않습니다. 당신이 1 차 미분의 지역 근사값이 추정 지점에서 아주 멀게 일반화되지 않을 수도 있는 지역에 있을 수 있기 때문에 (예 : 거대한 언덕 직전의 하향 곡선), 당신은 일반적으로 신중하길 원하고, 너무 큰 걸음(스텝 사이즈)를 설정하면 안 됩니다. 그래서 우리는 신중하게 아래의 방정식에서 스텝 사이즈 α로 우리의 진행 과정을 통제합니다.

![alpha](https://drive.google.com/uc?export=view&id=17DH2IWicKTijSAtfNev_efBK1vOhK3QT)

이 스텝 사이즈는 흥미로운 작업을 합니다. 즉, 그라디언트 방향으로 각 매개 변수를 업데이트하고 고정 된 크기만큼만 실행하도록 거리를 제한합니다.
이 알고리즘의 가장 단순한 버전에서는 상수(스칼라)인 알파를 취하고, (예를 들어 알파는 0.1) 이를 손실에 대한 그라디언트에 곱합니다.

생각해보면 우리의 그래디언트는 실제로 벡터로 모델에서 각 매개변수에 대한 손실의 그라디언트입니다. 따라서 벡터에 스칼라를 곱하면 우리는 유클리드 매개변수 거리의 관점에서 동일한 고정된 양만큼을 각 매개 변수 축을 따라 확대되도록 곱하는 업데이트를 하게 됩니다. 
또한 경사강하법의 가장 기본적인 버전에서는 학습 과정에서 동일한 스텝사이즈를 사용합니다.
하지만 ... 정말 말이 되나요? 학습률이 낮다는 전제는 그라디언트의 국부적인(local) 추정은 오로지 그 추정치 주변의 작은 국부적인 지역에서만 유효하다는 것을 뜻합니다. 

 그러나 매개 변수들은 서로 다른 크기(scale)로 존재할 수 있으며 당신의 학습된 조건부 분포에 각기 다른 영향을 미칠 수 있습니다. 그리고 이러한 영향력 정도는 훈련 과정에서 변동될 수 있습니다. 이러한 관점에서, 유클리드 변수 공간에서 고정된 전역 반경으로 안전 버블을 정의하는 것이 특히 현명하거나 의미있는 일이라고 보지 않습니다.

![](https://cdn-images-1.medium.com/max/600/1*jeMxJLZz-o5xniDMKqcTAg.jpeg)


자연 경사의 지지자에 의해 암묵적으로 만들어진 반박 제안은 매개변수 공간에서의 거리로 안전 창을 정의하는 대신 분포 공간에서의 거리로 안전 창을 정의해야 합니다. 따라서, "현재 벡터의 입실론 거리 내의 매개변수 벡터를 유지하는 조건으로 현재 그라디언트를 따라갈 것입니다"대신 "내 모델이 이전에 예측한 분포의 입실론 거리 내에서 예측하는 분포를 유지하는 조건하에 나의 현재 그라디언트를 따라갈 것입니다. "라고 말합니다.
여기에서의 개념은 분포들 사이의 거리들이 모든 스케일링이나 이동 또는 일반적인 재파라미터화에 대해 변하지 않는다는 것입니다. 예를 들어, 분산 매개변수 또는 스케일 매개변수 (1 / 분산)를 사용하여 동일한 가우스를 매개변수화 할 수 있습니다; 매개 변수 공간을 살펴보면 두 분산은 분산 또는 스케일을 사용하여 매개 변수화되었는지 여부에 따라 거리가 달라집니다. 그러나 원시 확률 공간에서 거리를 정의하면 일관성이 있습니다. 

이 글의 나머지 부분에서는 자연 그라디언트 학습 (Natural Gradient learning)이라는 접근법을 보다 강력하고 직관적으로 이해하려고 노력할 것입니다, 자연스러운 그라디언트 학습은 개념적으로 우아한 아이디어로, 매개 변수 공간에서의 임의적 비율을 수정하려고 하는 것입니다.
나는 그것이 어떻게 작동하는지, 그것을 만드는 다른 수학적 아이디어 사이에 어떻게 연결다리를 만드는지, 그리고 궁극적으로 그것이 실제로 유용한 지 그렇다면 어디에서 유용한 지에 대해 토론할 겁니다. 그러나, 첫째: 분포 사이의 거리를 계산한다는 것은 무엇을 의미합니까?

# Licensed to KL 

KL Divergence(발산) 또는 더 적절하게 Kullback-Leibler divergence는 기술적으로 분산들 사이의 거리 메트릭(거리, 측량)이 아닙니다 (수학자는 메트릭 또는 적절한 거리라고 불릴 수 있는 것에 대해 까다롭지만 [수학자들은 아무거나 메트릭이라 부르지 않는다는 뜻인듯]), 그러나 그것은 이 아이디어[분포들 사이의 거리]에 매우 근접합니다 .

![](https://cdn-images-1.medium.com/max/800/1*lMXoMVnzFZVW5CtcDZERew.png)

수학적으로, 한 분포 또는 다른 분포에서 샘플링 된 x의 값을 취한 로그 확률 (즉, 확률 값의 원시 차이) 비율의 추측된 값을 계산함으로써 KL 발산은 행해집니다. 기대(추측)가 분포들 중 하나 또는 다른 것으로부터 얻어진다는 사실은 그것(KL발산)을 비대칭 측정으로 만듭니다, 여기서 비대칭이란 KL (P || Q)! = KL (Q || P)입니다. 그러나 많은 다른 방법으로 KL 분산은 확률 거리가 어떻게 생겨야(정의되어야) 하는지에 대한 우리의 개념으로 맵핑됩니다.[KL 발산이 우리가 생각하는 확률 거리의 성질들을 만족할 것이다.] 즉, 확률 밀도 함수가 정의되는 방식을 기반으로 하여 그 분포가 정의되는 영역의 점들의 집합에서 밀도값의 차이를 직접 측정합니다. 이것은 매우 실용적인 측면을 가지고 있습니다. 그로 인하여 광범위한 X 시리즈(확률분포 X, X축 전체 말하는 듯)데 대하여 "이 X의 확률은 무엇입니까?"라는 질문에 대하여 분포들이  멀리 떨어진 답(많이 다를 때)을 하면 할수록 그 분포들이 더 다르다고 보여질 것이다. 

Natural Gradient의 맥락에서 KL 분산은 우리 모델이 예측하는 출력 분포의 변화를 측정하는 방법으로 사용됩니다. 다중 방향 분류 문제를 푸는 경우, 우리 모델의 출력은 다항식 분포로 볼 수있는 softmax가 될 것이고, 각 클래스에 다른 확률이 배치됩니다. 현재 매개 변수 값에 의해 정의 된 조건부 확률 함수에 대해 이야기 할 때, 이것은 우리가 말하는 확률 분포입니다. 우리가 그라디언트 스텝을 확장하는 방법으로 KL 분산을 사용한다면, 주어진 입력 집합에 대해 매우 다른 클래스 분포들의 예측을 유도한다면(서로 다른 클래스 분포들을 예측해 본다면) 이 공간에서 "더 멀리 떨어져 있는"두 매개 변수 형태가 있음을 의미합니다. 

# The Fisher Thing

지금까지 우리는 매개 변수 공간에서 업데이트 단계의 거리를 조정하는 것이 불만족스럽게 임의적이었던 이유를 논의하였고, 덜 임의적인 대안을 제안했습니다 : 우리 모델이 이전에 예측했던 클래스 분포로부터의 KL Divergence의 관점으로 최대 거리에서 일정 거리 만 이동하는 우리의  스텝을 조정합니다. 나에게 Natural Gradient를 이해하는 가장 어려운 부분은 다음에 나오는 부분입니다: KL Divergence와 Fisher Information Matrix 사이의 연결. 
이야기의 끝으로 시작하여 Natural Gradient는 다음과 같이 구현됩니다 : 
     ![](https://cdn-images-1.medium.com/max/1000/1*1YdL1eZim3--C9zHFNkY_g.png)

 등호 위의 def는 오른쪽에 있는 것이 왼쪽에있는 기호의 정의라는 것을 의미합니다. 오른쪽 용어는 두 부분으로 구성됩니다. 첫째, 손실 함수의 매개변수들에 대한 그라디언트가 있습니다 (이것은 보다 일반적인 그라디언트 디센트 단계에서 사용되는 것과 동일한 그라디언트입니다). "자연적"비트는 두 번째 구성 요소에서 비롯됩니다. 로그 확률 함수의 제곱 된 그래디언트의 z에서 취해진 예상 값입니다. 피셔 정보 매트릭스 (Fisher Information Matrix)라고 불리는 이 전체 객체를 가져 와서 우리의 손실 그래디언트에 그것의 역수(역행렬)를 곱합니다. 
 
 
p-theta (z) 항은 우리의 모델에 의해 정의된 조건부 확률 분포, 즉 말하자면 : 신경망 끝의 softmax입니다. 우리는 모든 p-theta 항들의 그라디언트 를보고 있습니다. 왜냐하면 우리는 예측 된 클래스 확률이 매개 변수 변경의 결과로 변경되는 양을 고려하기 때문입니다. 예측 된 확률의 변화가 클수록 우리의 pre-update와 post-update 예측된 분포 사이의 KL 발산이 커집니다.

Natural gradient 최적화 혼란을 만드는 부분은 당신이 그것을 읽거나 생각할 때, 당신이 이해하고 주장해야할 두 가지로 구별되는 그라디언트 객체가 있다는 것입니다, 즉 다른 것들이 있다는 뜻입니다. 덧붙이자면, 이것은 불가피하게 많은 일에 압도당하도록 할것이고, 특히 우도 도(likelihood)를 논의 할 때, 그리고 전반적인 직감을 파악할 필요가 없습니다. 당신이 모든 참혹한 세부 사항을 거치고 싶지 않으면 다음 섹션으로 건너 뛸 수 있습니다


# 손실에 관한 기울기 
![](https://cdn-images-1.medium.com/max/600/1*8HWxP8SpeAE-wT8Bpf6hRQ.png)

일반적으로 분류 손실은 교차 엔트로피 함수이지만 보다 광범위하게는 모델의 예상 확률 분포와 실제 목표 값을 입력하고 대상에서 멀리 떨어져있을 때 더 높은 값을 갖는 함수입니다. 이 객체의 그라디언트는 그라데이션 강하 학습의 핵심적인 필수요소입니다. 그것은 각 매개 변수를 한 단위 씩 이동하면 손실된 양이 변화하는 것을 보여줍니다.

# 로그 우도 (log likelihood)의 그래디언트

![](https://cdn-images-1.medium.com/max/600/1*XEuAR_Ju3Qg24BXQvEsFdA.png)

이것은 나에겐 natural gradient를 배우는 것에 있어서  가장 혼란스러운 부분이었습니다. 왜냐하면 Fisher Information Matrix에 관해 읽으면 모델의 로그 우도 (log likelihood)의 그래디언트와 관련이 있다고 설명하는 많은 링크를 얻을 수 있기 때문입니다. 우도 함수에 대한 나의 이전의 이해는 당신의 모델이 데이터의 일부 집합이라고 생각할 가능성이 얼마나 큰지를 나타내는 것이었습니다. 특히 목표를 계산하려면 목표 값이 필요했습니다. 왜냐하면 당신의 목표는 모델이 실제 목표에 할당할 확률을 입력 특징에 조건을 지정할 때 계산하는 것이었기 때문입니다. 가장 보편적인 Maximum Likelihood 기법과 같은 가능성이 논의되는 대부분의 상황에서는 우도가 높을수록 모델이 실제 분포에서 샘플링된 값을 할당할 확률이 높아지고 우리 모두가 행복해지기 때문에 로그 우도의 그라디언트에 신경을 써야합니다. 실제로는 데이타의 실제 클래스 분포로부터 만들어진 기대의 확률과 함께 p (class | x) 그래디언트의 예상 값을 계산합니다.

그러나 다른 방법으로 우도를 평가할 수도 있고, 실제 목표 값에 대한 우도를 계산하는 대신 (진짜 목표로의 확률을 높이기 위해 매개 변수를 푸시할 수 있기 때문에 0이 아닌 그라디언트가 예상되는 경우), 당신은 조건부 분포 그 자체에서 끌어 낸 확률을 사용하여 기대치를 계산할 수 있습니다. 즉, 네트워크가 softmax를 얻는 경우, 주어진 관측치에 대한 데이터의 실제 클래스를 기반으로 확률이 0/1 인 logp (z)를 기대하는 대신, 모델의 예상 확률을 기대에서 가중치로 사용하십시오. 이것은 우리의 모델의 현재 신념을 근거 진실로 제공하기 때문에 전반적으로 예상되는 그라디언트 0으로 이어질 것이지만 그래디언트의 분산 (즉, 그래디언트 제곱)의 추정치를 얻을 수 있습니다. 이는 피셔 매트릭스에서  (암시적으로)  예측 클래스 공간에서 KL 발산을 (암시 적으로) 계산하기 위해 필요한 것입니다.


# 그래서 ... 도움이 되나요?

이 게시물은 역학에 대해 많은 시간을 보냈습니다. 자연 그라디언트 추정기라고 정확히 부르는 것이 무엇이며, 어떻게 작동하는지, 왜 작동하는지에 대한 더 나은 직감은 무엇입니까? 그러나 내가 다음의 질문에 대답하지 않으면 나는 기분이 나빠질 것 같다.: 이게 실제로 가치를 제공합니까? 

짧은 대답은 실질적으로 말하자면: 대부분의 심층 학습 응용 프로그램에 공통적으로 사용될 수 있는 충분한 가치를 제공하지 못합니다. natural 그라디언트가 적은 단계로 수렴하는 증거가 있지만, 나중에 설명 하겠지만 그것은 다소 복잡합니다. natural 그라디언트 개념은 능력이  좋고 매개 변수 공간에서 업데이트 단계를 조정하는 임의적인 방법으로 인해 좌절한 사람들을 만족시킵니다. 그러나 능력이 좋지 않고, 더 많은 경험적 방법을 통해 제공될 수 없는 가치를 제공한다는 것이 분명하지 않습니다.

내가 알 수있는 한, natural 그라디언트는 두 가지 핵심 가치 원천을 제공합니다.


#### 1. 곡률 에 대한 정보를 제공합니다.
#### 2. 그것은 손실 공간에서 모델의 움직임과는 별도로 예측된 분포 공간에서 모델의 움직임을 직접 제어하는 방법을 제공합니다.

# 곡률 

현대 그라데이션 강하의 놀라운 점 하나는 1차 방법으로 완성 된 것입니다. 1차 방법은 2차 미분이 아닌 당신이 업데이트하려는 매개 변수에 대한 미분만을 계산하는 방법입니다. 1차 도함수를 사용하면 특정 지점에서 곡선에 대한 접선(다차원 버전)만을 알 수 있습니다. 그 접선이 얼마나 빨리 변하는지 알지 못합니다 : 2차 미분 또는 더 구체적으로 말하자면, 주어진 방향에서 함수가 가지는 곡률 수준. 기울기가 극단적으로 변화하는 높은 곡률의 지역에서 가파른 산을 오르 내리는 신호가 당신을 도망치지 않도록 큰 스텝을 잡는 것을 조심해야 하므로 곡률은 알면 유용합니다. 이것에 대해 내가 생각하고 싶은 방법(엄격한 것보다 더 발견적인) 은 만약 당신이 점마다 그래디언트가 가변적인 지역에 있다면 (예를 들어, 높은 분산), 당신의 그래디언트 미니배치 추정은 어떤 의미에서는 더 확실하지 않다는 것입니다. 대조적으로, 그라데이션이 주어진 지점에서 거의 변경되지 않는 경우 다음 단계를 수행할 때 주의를 기울이지 않아도 됩니다. 2차 미분 정보는 곡률 수준에 따라 스텝을 조정할 수 있으므로 유용합니다. 

Natural 그라디언트가 실제로 기계적으로 수행하는 작업은 매개 변수 업데이트를 그라디언트의 2차 미분으로 나누는 것입니다. 주어진 매개 변수 방향에 대해 그라디언트가 변경 될수록 Fisher Information Matrix의 값이 높아지며 해당 방향의 업데이트 스텝이 더 낮아집니다. 문제의 그래디언트은 당신의 배치(batch)에서 점들에 대한 경험적 우도의 그래디언트입니다. 손실에 대한 그라디언트와 같은 것은 아닙니다. 그러나 직관적으로, 우도가 극적으로 변하는 것이 손실 함수의 극적인 변화와 일치하지 않는 것은 드물 것입니다. 주어진 점에서 로그 우도-미분 공간의 곡률에 대해 정보를 캡처함으로써 Natural Gradient는 실제의 기본 손실 공간에서 곡률에 대한 좋은 신호를 제공합니다. Natural Gradient가 수렴 속도를 높이는 것으로 나타났을 때 (적어도 필요한 그래디언트 단계 수와 관련하여) 이점이 어디서 발생했는지에 대해 매우 강력한 주장이 있습니다.

그러나. 그라디언트 단계에서 Natural Gradient가 수렴 속도를 높이는 것으로 나타났다는 것을 주목하십시오. 이 정확도는 Natural Gradient의 각 단계마다 n_parameters² 공간에 존재하는 양이라는 Fisher Information Matrix를 계산해야하므로 더 오래 걸립니다. 이러한 극적인 감속은 실제로 진정 손실 함수의 2차 미분을 계산함으로써 유발된 감속과 유사합니다. 그것이 사실일지도 모르지만 Natural 그라디언트 피셔 행렬을 계산하는 것이 기본 손실에 대한  2차 미분값을 계산하는 것보다 빠르다는 것을 어느 곳에서도 언급된 것을 보지 못했습니다. 이를 가정 할 때, 손실 자체에 대해 직접적인 2차 미분 최적화를 수행하는 것과 비교할 때 자연 그레디언트 (Natural Gradient)가 제공하는 미미한 값을 알기는 어렵습니다. 

현대 신경 네트워크가 성공할 수 있었던 이유는 1차 미분만의 방법은 실패 할 것이라고 이론이 예측할 수 있었던 이유는 딥러닝 실무자가 분석적 2차 미분 행렬에에 포함되어질 정보를 근본적으로 실험적으로 근사화하는 영리한 속임수를 발견했다는 것입니다. 

- Momentum은 최적화 전략으로서 지난 그래디언트 값들을 기하급수적으로 가중 평균하여 유지하고 주어진 그래디언트 업데이트를 지난 이동 평균으로 바이어스하여 작동합니다. 이것은 그래디언트 값이 격렬하게 변화하는 공간의 일부에 있는 문제를 해결하는 데 도움이 됩니다: 모순된 그래디언트 업데이트를 지속적으로 접하는 경우, 그들은 당신의 학습 속도를 늦추는 것과 유사한 한 가지 또는 다른 강한 의견을 가지지 않고 평균을 냅니다. 반대로, 같은 방향으로 진행되는 그래디언트 추정치를 반복적으로 얻는다면, 이는 낮은 곡률 영역을 나타내는 것이며, 모멘텀은이를 따르는 큰 스텝의 접근을 제안합니다. 

- RMSProp은 재미있게도 Geoff Hinton 중반 Coursera 과정에서 고안한 것으로 Adagrad라는 기존 알고리즘을 약간 수정한 것입니다. RMSProp은 지난 제곱된 그래디언트 값의 기하급수적 이동 평균, 즉 그라디언트의 과거 분산을 취하여 해당 값으로 업데이트 스텝를 나눔으로써 작동합니다. 이는 대략 그라디언트의 2 차 미분의 경험적 추정으로 생각할 수 있습니다. 

- Adam (적응 형 모멘트 추정)은 본질적으로 이러한 두 가지 접근 방식을 결합하여 그라디언트의 실행 평균 및 실행 분산을 모두 추정합니다. 가장 일반적이며 가장 많이 사용되는 최적화 전략 중 하나입니다, 왜냐하면 그것은 주로 노이즈가 많은 1차 미분 그래디언트 신호가 될 수있는 것을 부드럽게하는 효과가 있기 때문입니다.

재미있는 점과 이 모든 접근법에 대해 언급 할 가치가있는 점은 함수 곡률 측면에서 업데이트 스텝을 일반적으로 확장하는 것 외에도 특정 방향의 곡률에 따라 업데이트 방향을 다르게 조정한다는 것입니다. 이것은 이전에 논의했던 것과 같은 것으로, 같은 양으로 모든 매개 변수를 조정하는 것이 합리적인 일이 아닐 수도 있습니다. 거리의 곡률이 높으면 유클리드 매개 변수 공간의 동일한 단계에서 한 단계가 예상되는 그라디언트 값의 변화 측면에서 더 멀리 이동할 수 있습니다. 

따라서 매개 변수 업데이트에 대한 일관된 방향을 정의 할 때 자연스러운 그라디언트의 우아함이 없지만 대부분 동일한 상자를 확인합니다: 방향을 바꾸거나 곡률이 다른 시점, 그리고 개념적으로 주어진 크기의 매개 변수 단계가 실제 영향 정도가 다른 지점에서 업데이트 단계를 적용하는 기능.

# 직접 분배 제어 Direct Distributional Control

그렇습니다. 마지막 절에서 논의한 바에 따르면, 우리의 목표가 로그 우도의 분석 곡률 추정치를 손실의 곡률 추정치의 대립으로 사용하는 것이라면, 두 가지 분석적 N² 계산 모두 상당히 많은 시간이 소요되는 것으로 나타났으니 후자를 수행하거나 후자를 근사해 보는 것이 어떻겠습니까? 그러나 당신이 실제로 손실의 변화에 대한 대변자가 아닌 예측 된 클래스 분포의 변화를 실제로 염려하는 상황에 있다면 어떻게 해야 할까요? 그런 상황이 어떻게 생겼을까요? 

![](https://cdn-images-1.medium.com/max/600/1*5dtG0VF_pAzx_aPCQgNFGQ.png) 

이러한 상황의 한 예가 아마도 우연히 자연 그라디언트 접근법의 현재 사용되는 주요 영역 중 하나일 것입니다: 보강 학습 영역에서 신뢰 영역 정책 최적화. TRPO의 기본 직감은 치명적인 실패 또는 파국적인 붕괴라는 아이디어에 싸여 있습니다. 정책 그라디언트 설정에서 모델의 끝에서 예측하는 분포는 일부 입력상태에 따라 조건부로 동작하는 분포입니다. 그리고 정책에 대해 배우고 있다면 다음 훈련 라운드의 데이터가 모델의 현재 예상 정책에서 수집되는 곳에서 더 이상 흥미로운 데이터(예를 들어, 원을 그리며 돌아 다니며 배울 수있는 유용한 보상 신호를 얻지 못할 수도 있습니다)를 수집 할 수 없는 영역으로 정책을 업데이트 할 수 있습니다. 이것은 정책이 파국적 인 붕괴를 겪는 것을 의미합니다. 이를 피하기 위해 주의를 기울여야 하며 정책을 크게 변경하는 그래디언트 업데이트는 수행하지 말아야 합니다 (특정 시나리오에서 서로 다른 작업에 적용할 확률에 관한). 우리가 예측 된 확률을 얼마만큼 변화시킬 수 있는지에 관해서 조심스럽고 점진적이라면 갑작스럽게 실행 불가능한 체제로 뛰어 넘을 수 있는 능력이 제한됩니다.

다음은 Natural 그레디언트보다 강력한 사례입니다: 여기에서 우리가 통제하는 실제 사항은 새 매개변수 구성에서 예상되는 다양한 동작의 확률이 얼마나 달라지는 지입니다. 그리고 우리는 손실 함수에 대한 대리변수가 아니라 그 자체로도 중요합니다. 

# Open Questions

설명자 글의 프레임이 완전한 이해력의 아주 높은 위치를 암시하지만 실제로는 그렇지 않기 때문에 주제에 대해 아직도 혼란스러운 부분이 있다는 것을 알려주면서 이 게시물을 마무리하고 싶습니다. 언제나처럼, 내가 잘못 생각한 것을 눈치 채면, / DM에게 의견을 말하십시오. 나는 그것을 바로 잡을 것입니다!

- 로그 가능성을 계산하는 것이 피셔 매트릭스가 손실 함수의 헤시안을 계산하는 것보다 더 효율적인지 여부를 결코 결정적으로 알지 못했습니다 (Natural 그라디언트가 손실 표면에 대한 곡률 정보를 얻는 더 저렴한 방법이라는 주장이 될 것입니다)

- 비교적 가능성이 있지만 로그 확률 z에 대한 예상 값을 취하면 기대치가 우리 모델에 의해 예측된 확률보다 더 많이 취해지는 것을 완전히 확신하지는 않습니다 (기대 값은 확률과 관련하여 정의해야합니다).
