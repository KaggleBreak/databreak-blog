---
layout: post
title: "Combining multiple feature selection methods for stock prediction Union, intersection, and multi-intersection approaches Review (KR)"
image: ./img/RyanYS/2019-03-28-Combining_feature_selection.png
author: RyanYS
date: "2019-03-28T23:46:37.121Z"
tags:
  - FeatureSelection
---

## 참고자료
- 원본 아티클 (Combining multiple feature selection methods for stock prediction Union, intersection, and multi-intersection approaches Review)
    - [Combining multiple feature selection](https://xhxt2008.live/2017/12/21/Combining-feature-selection/)
    - 타이틀과 같은 이름의 논문을 리뷰한 아티클

## Abstract
투자자의 주가를 효과적으로 예측하는 것은 매우 중요한 연구 문제입니다. 문헌에서 데이터 마이닝 기술은 주식 (시장) 예측에 적용되었습니다. 데이터 마이닝의 사전 처리 단계인 피쳐 선택은 효과적인 예측을 위해 주어진 데이터 세트에서 대표성이 없는 변수를 필터링하는 것을 목표로 합니다. 서로 다른 피쳐 선택 방법을 사용하면 서로 다른 피쳐가 선택되어 예측 성능에 영향을 미치므로 이 논문의 목적은 더 나은 예측을 위해 더 많은 대표 변수를 식별하기 위해 여러 피쳐 선택 방법을 결합하는 것입니다.

## Introduction
주식 투자는 전 세계적으로 매우 인기 있는 투자 활동입니다. 문헌에서 재무 비율, 기술 지표 및 거시 경제 지표와 같은 몇 가지 기본 요소가 주가 상승 및 하락에 영향을 미치는 중요한 요인으로 입증되었습니다. 그러나 다른 연구에서는 예측 모델에 대해 그들의 요인 (즉, 입력 변수)을 다르게 선택합니다. 즉, 가장 대표적인 변수가 무엇인지에 대한 정확한 답이 없기 때문에 주식 예측의 중요한 요소에 대한 의견은 관련 연구에서 다소 차이가 있습니다. 다른 한편으로, 다른 입력 변수를 사용하면 동일한 예측 모델이 다르게 수행 할 수 있다는 사실이 있습니다. 따라서 투자자를 위한 최적의 주식 예측 모델을 구축하는 것은 매우 어렵습니다. 
선택된 데이터 세트에서 중복 또는 무의미한 피쳐를 필터링하여 더 나은 예측 성능을 위해 보다 대표적인 피쳐를 얻을 수 있습니다. 
즉, 선택된 피쳐 선택 방법은 주가 예측을 위해 사용 가능한 피쳐를 선택해야합니다.그러나 다른 피쳐 선택 방법을 사용하면 다른 결과가 발생할 수 있습니다.
  따라서 여러 가지 피쳐 선택 방법을 적용한 다음 선택 결과를 결합 할 수 있다면 모든 피쳐 선택 방법이 동의하는 가장 중요하고 대표적인 변수를 이해할 수 있을 뿐만 아니라 하나의 피쳐 선택 방법을 사용하는 것보다 예측 성능을 향상시킬 수 있습니다.

## Related Work
![Related Work](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkepwajznj215o0gy4d1.jpg)

## Literature review
### Stock price theory
- 주가는 시장의 매수자와 매도인을 통한 실제 거래 가격을 의미합니다. 주가는 수요와 공급의 법칙에 의해 결정됩니다. 이론적으로 주식 가격이 높든 낮든 간에 공개 시장에서의 매수인과 매도인의 거래가 결정됩니다. 수요와 공급이 바뀌면 주가가 바뀌어야 합니다.
- [파마의 효율적인 시장 가설](https://en.wikipedia.org/wiki/Efficient-market_hypothesis)은 투자 활동이 "공정한 게임 시장"이라고 가정합니다. 그것은 모든 정보가 주식 시장에 공개되고 주식 가격에 반영되었음을 의미합니다. 공개된 정보의 차이에 따라 효율적 시장 가설에는 3가지 종류가 있습니다.
    - 약한 형태의 효율적인 시장
    - 세미 폼 (Semi-strong Form) 효율적인 시장
    - 강력한 형태의 효율적인 시장

### Stock price analysis methods
- **기본 분석** : 기본 분석은 모든 주식이 본질적인 가치를 가지고 있다고 믿습니다. 주가가 내재 가치보다 낮으면 주가가 저평가되어 있음을 의미합니다. 또한 경제 요소도 이 범주에 속합니다.

- **기술 분석** : "차트 작성"이라고도 하는 기술적 분석은 수십 년 동안 금융 관행의 일부였습니다. 미래 가격 움직임을 예측하기 위한 주요 도구로 차트를 사용하여 주가의 역사적 가격과 물량 움직임을 연구합니다. 이 이론은 투자 수단의 가격, 거래량, 폭 및 거래 활동의 경향과 패턴이 의사 결정자가 그 가치를 결정하기 위해 활용할 수 있는 관련 시장 정보의 대부분을 반영한다고 믿습니다.

### Feature Selection
패턴 인식과 같은 많은 연구 문제에서 더 많은 예측 정보를 가진 속성 집합의 그룹을 선택하는 것이 중요합니다. 즉, 관련이 없거나 중복된 피쳐의 수를 크게 줄이면 학습 알고리즘의 실행 시간도 줄어듭니다. 또한, 보다 일반적인 개념이 산출 될 수 있습니다. 피쳐 선택을 수행하면 데이터 시각화 및 데이터 이해를 용이하게 하고 Storage requirement를 줄이며 Training 및 예측 시간을 줄이고, 차원의 저주를 피해 예측 성능을 향상시키는 등 많은 잠재적 이점을 얻을 수 있습니다.

### PCA
주성분 분석 (PCA)은 다변량 통계 기법입니다. 상호 연관 변수가 많은 데이터 세트의 차원을 줄이는 것을 목표로합니다. 특히, 원래의 특징을 유지하면서 상관 관계가 높은 요소로 구성된 요소 또는 구성 요소의 작은 집합을 추출합니다. PCA를 수행 한 후에 component라는 uncorrelated 변수가 원래 변수를 대체합니다.
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkenyuw81j20ts0dhjsr.jpg)

### GA-SVM
유전자 알고리즘 (Genetic Algorithms, GA)의 주요 개념은 적자 생존에서의 자연 선택으로부터 진화론에 대한 다윈의 이론으로부터 나온 것입니다. GA는 자연 선택이 작용하는 과정을 계산적으로 모방하려고 시도합니다.
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkenzjpnjj20fr0c8aaw.jpg)

### Classification and Regression Trees (CART)
분류 및 회귀 나무 (CART)는 많은 설명 변수 중에서 설명할 응답 변수를 결정할 때 가장 중요한 변수를 선택할 수 있는 통계 기법입니다. CART에 의해 생성 된 의사 결정 트리는 엄격하게 2진이며 각 의사 결정 트리마다 정확히 2개의 분기를 포함 합니다. 루트 노드 t는 어떤 조건을 기반으로 두 개의 샘플로 분리됩니다. 조건에 맞는 샘플은 왼쪽 노드 (tl) 로 분리되고 나머지는 오른쪽 노드 (tr) 로 분리됩니다. 
특히, 결정 트리는 엔트리 이론에 기반하여 가장 높은 정보 이득 (또는 최대 엔트로피 감소)을 갖는 속성 (또는 특징)이 non-leaf 노드에 대한 테스트 속성으로 선택됩니다.


Related Work는 하나의 선택한 피쳐 선택 방법을 적용하여 관련 없는 변수를 필터링합니다. 이는 우리가 문헌의 주식 예측에 사용된 모든 관련 변수를 모으고 여러 가지 피쳐 선택 방법을 결합하여 예측 성과를 개선하기 위한, 보다 대표적인 변수를 식별하도록 동기를 부여합니다.

## Experimental design
### The first experimental stage
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkenz2fsrj208o0a4gmb.jpg)

### The second experimental stage
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkeo0f4ppj20a60br0tm.jpg)

## Data Set
### First set
![](http://drive.google.com/uc?export=view&id=15izf93vplvYVZZXaBn2GcA9WVcCugPyY)

### Second set
![](http://drive.google.com/uc?export=view&id=18dcik7oRqpIkBXoIeic9gLWd4FEs-xhK)

## Features
![](http://drive.google.com/uc?export=view&id=1nHfnyJxqQtypEE03Yv8_WOQxmo45kbkf)

## Results
### Single feature selection methods
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkeodedmdj20q709fjva.jpg)

![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkeoerrm1j20q308kn0g.jpg)

첫 번째, 두 번째 데이터 세트를 기반으로 네 가지 다른 MLP 모델의 예측 정확도를 보여줍니다. 우리가 볼 수 있듯이, 다른 테스트 데이터 세트가 고려되면 결과는 약간 다릅니다. 다중 피쳐 선택을 한 모델이 분명히 우수하다는 것을 알 수 있습니다. 그리고 그들은 비슷한 효과를 가집니다.

### Multiple feature selection methods
![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkeoepkscj20q50acdk9.jpg)

![](http://ww1.sinaimg.cn/large/6b8ee255gy1fxkeoczzt6j20q709ntd0.jpg)

결과는 PCA와 GA의 교집합이 첫 번째 데이터 집합에서 다른 조합 접근법보다 우월함을 나타냅니다. 다른 한편으로, 다중 교집합 접근법에 의한 다중 피쳐 선택 방법의 결합은 두 번째 데이터 세트에 최선의 결과를 냅니다. 그러나 두 번째 테스트 데이터 세트에서 다중 교집합과 PCA와 GA의 교집합의 예측 정확도는 큰 차이가 없습니다. 즉 0.3 % 미만입니다.

## Conclusion
- 이 논문에서는 PCA (Principal Component Analysis), GA (Genetic Algorithms) 및 CART (Decision Tree)와 같은 3가지 피쳐 선택 방법을 비교하고 결합 정확도 및 오류를 검사하기 위해 결합, 교집합 및 다중 교집합 접근 방식을 기반으로 결합합니다.
- 실험 결과는 다중 피쳐 선택 방법을 결합하는 것이 단일 피쳐 선택 방법을 사용하는 것보다 더 나은 예측 성능을 제공할 수 있음을 보여줍니다. 특히, PCA와 GA의 교집합과 PCA, GA 및 CART의 다중 교집합이 가장 잘 수행되어 예측 정확도가 가장 높고 예측 주가가 가장 낮은 오류율을 제공합니다.
- 더욱이, 이 두 가지 결합된 접근법은 많은 비 대표 변수를 걸러내어 85개의 원래 변수에서 각각 14개와 17개의 중요한 변수를 선택합니다. 이러한 변수는 실질적인 투자 결정에 사용될 수 있습니다.