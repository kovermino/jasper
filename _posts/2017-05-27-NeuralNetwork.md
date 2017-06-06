---
layout: post
cover: false
title: Neural Network
date:   2017-05-27 10:00:00
tags: fiction
subclass: 'post tag-fiction'
categories: 'Machine Learning'
---

어떤 기술을 더 깊이 이해하기 위해서는 어떻게 하는 것이 좋을까요? 가장 좋은 방법은 당연히 실제로 그 기술을 사용해서 뭔가를 만들어보는 것입니다. 실제로 움직이는 프로그램을 밑바닥부터 만들며 그 소스코드를 읽고 고민해보는 과정은 어떠한 기술을 이해하는 데 아주 중요하기 때문입니다. 그러므로 딥러닝에 대해 이해하기 위해서는 인공신경망의 원리를 먼저 깊이 이해하고, CNN과 RNN을 학습하는 순서로 가야한다고 생각했습니다. 물론 개념자체는 한번씩 보았던 내용이지만 사람인지라 잊어버린 내용들을 한 번씩 되짚어주는 과정이 필요하다는 생각에 신경망, 즉 Neural Network에 대해 포스팅하게 되었습니다.

이번 포스팅은 Neural Network의 기본원리와 용어들, 그리고 파이썬으로 하나하나 구현한 코드들을 통해 전개하겠습니다. 그러나 너무나 기초적인 내용들은 가급적 다루지 않을 생각입니다. 미분을 하기위해서 덧셈부터 공부하려고 하면 시간이 너무 많이 들기 때문입니다. 그러나 논리의 흐름은 끊어지지 않게끔 차분히 설명할 예정입니다.

Neural Network를 사용한 딥러닝은 훌륭하다고 알려져있는 머신러닝 알고리즘들 중 하나입니다. 딥러닝이 포함된 개념인 머신러닝은 데이터 중심의 사고방식을 의미하죠. 즉 주어진 데이터를 활용하여 인식 알고리즘을 구현하는 방법론이라고도 볼 수 있습니다. 머신러닝은 보통 다음과 같은 단계로 진행됩니다.  

1. 특징(feature) 추출
2. 특징으로부터의 패턴(pattern) 발견
3. 학습(training)
4. 인식(classification)  

특징은 보통 벡터로 표현됩니다. 수치화된 벡터들의 집합를 통해 패턴을 인식하는 과정을 우리는 학습이라고 부릅니다. 잘 동작하는 머신러닝 알고리즘의 설계를 위해 우리는 범용능력이라는 것을 평가해야합니다. 즉 70000개의 데이터가 있다면, 그 중 60000개는 시스템을 학습시키는데 사용하고, 나머지 10000개를 통해 알고리즘이 학습한 케이스 외에 새로운 데이터에도 예측을 적용할 수 있는지 평가를 해야한다는 이야기입니다. 학습을 위해 사용한 데이터는 training set, 평가를 위해 사용하는 데이터는 test set이라고 부릅니다.

그러나 신경망은 일반적인 머신러닝 알고리즘과는 한가지 다른 면이 있습니다. 특징추출을 사람이 하지 않는다는 것이죠. 데이터에 녹아있는 패턴을 그대로 받아들여서 온전히 기계가 학습하도록 하는 알고리즘인 것입니다. 이러한 학습방식을 종단간 기계학습(end-to-end machine learning)이라고도 합니다.

'밑바닥부터 시작하는 딥러닝'에는 외부라이브러리 사용을 최소화한 딥러닝 프로그램의 소스코드와 구현원리가 잘 나와있습니다. 이 책에는 이론 설명과 파이썬 구현 코드라는 투 트랙으로 학습을 진행하다보면 여러가지 실험을 통해 기술을 본인의 것으로 흡수할 수 있을 것이라고 씌여있습니다.

신경망 이론에 자세하게 들어가기 전에 제가 학습했던 내용을 바탕으로 간략한 신경망 이론을 정리해보도록 하겠습니다.

## 퍼셉트론

신경망은 대표적인 지도학습(supervised learning) 알고리즘입니다. 인간의 뇌가 학습하는 방식에 기초하여 만들어진 방법인데, 사람의 뇌에서 뉴런이 하는 역할을 인공신경망에서는 퍼셉트론이라는 논리 게이트 형태로 치환했습니다. 컴퓨터공학의 논리 게이트는 퍼셉트론의 일종입니다만 그 자체는 아닙니다. 퍼셉트론은 논리게이트를 확장한 개념입니다. 즉 입력이 있고 출력이 있는 노드이며, 여기서 입력(신호)는 여러개가 될 수 있지만 출력값은 하나로 정해집니다.

간단하게 보여줄수 있는 예는 AND 게이트입니다. AND 게이트는 2개의 입력 신호를 받아 하나의 출력을 내는데, 입력되는 두개의 신호에 따라 다른 출력값을 냅니다. 아래 그림과 같죠.  

![](assets/images/andgate.JPG)

OR 게이트나 NAND 게이트도 마찬가지입니다. 두 개의 입력 신호값에 의해 하나의 출력값을 냅니다. 이러한 논리게이트들은 그 입출력값이 0과 1로 정해져있습니다. 전기회로를 설계하기 위한 목적으로 고안된 개념이기 때문이죠. 그렇다면 1과 0의 모든 조합에 대해서 특정 출력을 하도록 설계할 수도 있을까요? XOR 게이트를 생각해보면 알 수 있습니다. XOR 게이트를 하나의 퍼셉트론으로 구현할 수 있을까요?

그에 대한 대답은 '아니오'입니다. XOR 게이트는 하나의 논리게이트로는 구현이 불가능합니다. 왜일까요? 수학적으로 표현하면 XOR 게이트의 출력값은 선형분리불가능하다고 이야기합니다. 아래 그래프를 보시면 좀더 쉽게 이해할 수 있을 것입니다. 이것이 신경망이론이 마주했던 하나의 장벽이었습니다. 

![](assets/images/andxor.JPG)


그럼 XOR 게이트는 정말 구현이 불가능한 것일까요? 우리는 그렇지 않다는 것을 알고 있습니다. 우리가 원하는 XOR 게이트의 출력값을 내려면 AND, OR, NAND 게이트를 조합하면 됩니다. 이것이 다층 퍼셉트론의 개념입니다.

![](assets/images/xor.JPG)

그러나 신경망에서 사용하는 퍼셉트론은 이렇게 간단하기만 한 구조는 아닙니다. 입력값에 따른 출력값을 계산하기 위해 예측(forward propagation)에서는 가중치와 활성함수를 정해줘야하기 때문입니다. 학습을 위해서는 손실함수도 필요하죠. 연산을 빠르게 하기 위해서 역전파(backword propagation)의 개념도 필요합니다. 각각의 기법들을 피팅하기 위해서 gradient descent, 모멘텀, AdaGrad, Adam 등의 수학적 최적화 기법들의 개념도 알고 있어야합니다. 이 포스팅에서 이 모든 것을 다루지는 않을 것입니다. 여기에서는 mnist 데이터를 이용하여 학습하는 단순한 2층 신경망을 구현한 코드를 보면서 오류역전파까지만 소개할 예정입니다. 그리고 더 깊은 내용은 이후에 CNN(Convolutional Neural Network)를 포스팅하면서 다룰 생각입니다.

## 신경망 구조

신경망에서의 퍼셉트론은 게이트와 구조는 비슷하지만 처리되는 과정은 매우 다릅니다. 특징벡터와 가중치의 행렬연산이라는 구조는 비슷할지언정 활성함수의 종류와 연산속도를 위한 역전파 사용등에서 차이를 보이는 것이죠. 정말 간단한 형태의 신경망 구조를 한 번 보겠습니다.  

![](assets/images/twoLayerNet.JPG)

굉장히 간단한 구조의 신경망입니다. 색깔별 노드의 종류는 다음과 같습니다.  

- 파랑: 입력층
- 초록: 은닉층
- 빨강: 출력층
- 보라: 편향

이 그림에서는 입력층의 노드가 2개입니다. 즉 입력층은 (x1, x2)꼴의 벡터의 집합입니다. 물론 우리는 이렇게 벡터하나하나에 대해서 연산을 수행하지는 않지만 쉬운 이해를 위해 우선은 벡터의 집합으로 생각하겠습니다. 그리고 (x1, x2)의 입력은 신경망으로 들어와 다음 층의 각 노드들에 대해 연산을 수행하도록 설계되어있습니다. 즉, 신경망은 (x1, x2)를 받아 1층(첫번째 은닉층) 노드의 가중치를 노드별로 각각 곱해줍니다. 그럼 각각의 노드에 대해서 a값이 계산될 것입니다. 그리고 a값을 정의역으로 하여 활성함수 값을 구합니다. 다음 그림과 같습니다.  

![](assets/images/twoLayerNet2.JPG)

a1(1)노드에서의 연산을 볼까요? (x1, x2)의 입력을 받아서 우리는 첫번째 가중치 행렬을 곱하고 편향인 b1을 더해 a 값을 구합니다. 그리고 a에 대한 활성함수 값도 구해줍니다.

![](assets/images/anode.JPG)

a 노드의 출력값인 y는 곧 다음 층 노드의 입력값이 됩니다. 행렬식으로 표현하면 XW(1)+B(1)=A(1)이 되는 것이죠. 이처럼 노드에서는 행렬연산과 더불어 함수계산이 이루어집니다. h(x)로 표현된 함수를 우리는 활성함수(activation function)라고 부릅니다. 활성함수의 종류는 알고리즘 설계자가 선택하는 것으로 예전에는 시그모이드(sigmoid; 로지스틱함수)를 사용했지만 gradient vanishing 등 몇 가지 문제점들로 인해 최근에는 ReLU(Rectified Linear Unit) function을 주로 이용합니다.

그렇다면 출력층은 어떻게 설계해야 할까요? 우선 출력층의 노드 숫자를 결정해야합니다. 만약 딥러닝으로 분류문제를 해결하려한다면 분류하고 싶은 클래스의 수로 설정하는 것이 일반적입니다. 그리고 출력층은 은닉층과 다른 활성함수를 사용합니다. softmax function인데요, 이는 출력층 결과값들의 총 합을 1로 정규화해줌으로써 각 출력값을 해당 클래스로 분류될 확률로 생각할 수 있도록 도와줍니다. 단순히 softmax를 직접 구현하다보면 오버플로 문제에 부딫힐 수 있어 수식 그대로 코딩하지 않고 적절히 변형하여 사용해야합니다. 이는 파이썬으로 구현된 소스 코드를 살펴볼때 다시 이야기하도록 하겠습니다.

이렇게만 이야기하면 잘 와닿지 않을 수 있다는 생각이 들었습니다. 이렇게 생각을 해보죠. 만약 우리가 다음과 같은 데이터셋을 가지고 있다고 생각해봅시다.  

| x1 | x2 | class |
|----|----|-------|
| 2  | 3  | 0     |
| 3  | 1  | 0     |
| 4  | 2  | 0     |
| 5  | 4  | 2     |
| 5  | 6  | 3     |
| 4  | 6  | 2     |
| 9  | 3  | 2     |
| 7  | 5  | 3     |
| 1  | 6  | 0     |  

새로운 데이터: (5,7)

우리는 지금 학습과정을 공부하지는 않았습니다. 앞에서 신경망은 특징추출을 사람이 하지 않는다고 했었습니다. 데이터에 녹아있는 패턴을 그대로 받아들여서 온전히 기계가 학습하도록 하는 알고리즘이라고 이야기했었죠. 이 데이터만으로 본다면 이 모형은 (x1,x2)에 대해서 (0,2,3) 셋 중에 하나로 분류해줄수 있는 신경망이 됩니다. 만약 위 데이터로 적절히 학습되어서 모든 w값들과 b값들이 정해져있다고 가정해봅시다. 이 상황에서 (5,7)이 들어옵니다. 그럼 신경망 모형은 이 것을 (0,2,3) 이 셋중에 하나로 분류할 것입니다. 정확히 말하면 이 세개의 클래스일 확률을 출력해줄겁니다. (0.1,0.2,0.7) 이런 식으로 말이죠. 만약 신경망이 이런 결과를 낸다면 우리는 0.7, 즉 3일 확률이 70%라고 해석할 수 있으므로 이것은 3일 것이다라는 예측을 할 수 있게 되는 것입니다. 이러한 신경망의 추론과정을 순전파(forward propagation)라고 부릅니다.

## 신경망의 학습(역전파: backward propagation)

신경망을 구현할때, 데이터의 학습으로 얻어지는 결과는 무엇일까요? 학습을 한다는 것은 어떤 의미인가요? 결국 우리는 최적의 W와 B를 찾아야 합니다. 그러나 실제로 현업에서 유의미한 의미를 끌어내는 신경망은 수많은 가중치(w)와 편향(b)를 가집니다. 레이블링 된 데이터셋을 입력한 후에 그저 학습을 실행하는 것만으로 신경망이 최적의 W와 B를 찾게 하는 알고리즘은 어떤 방식으로 동작해야 할까요?

일단 우리는 초기값을 설정해줄겁니다. W와 B에 임의의 초기값을 넣어주는 것이죠. 신경망 학습에서 특히 중요한 것이 가중치의 초기값입니다. 가중치의 초기값을 무엇으로 설정하느냐가 신경망 학습의 성패를 가르는 일이 실제로 자주 있습니다. 때문에 연구를 통해 증명된, 실제로 권장되는 초기값 설정방법들이 있습니다.

가장 쉽게 떠올릴수 있는 방법은 초기값을 모두 0으로 주는 것입니다. 그러나 이것은 매우 나쁜 아이디어입니다. 이후에 설명하는 오류역전파를 적용한 신경망 모형에서는 초기값을 0으로 하면 학습이 이루어지지 않습니다. 초기값을 설정하는 방법은 활성함수의 종류에 따라 달라지지만 흔히 적용되는 방법은 다음 두 가지입니다.  

1. 표준편차가 0.01인 정규분포를 따르는 랜덤값 (ex. 0.01 x np.random.randn(10,100))
2. Xavier 초기값  

그러나 활성함수로 ReLU function을 사용할때는 조금 다릅니다. ReLU function을 이용할 때는 ReLU에 특화된 초기값을 사용하는 것이 권장됩니다. 이 특화된 초기값은 찾아낸 사람인 카이밍 히(Kaiming He)의 이름을 따서 He 초기값이라고 부릅니다. 많은 실험들을 통해 검증된 초기값들이기 때문에 현재 상황에서는 혁신적인 논문이 나오기 전까지 해당 초기값을 사용하여 학습을 시작하는 것이 좋을 것 같습니다.

초기값이 정해졌다고 하면, 실제로 데이터를 통해 학습하는 알고리즘은 어떻게 구현해야 할까요? 우선 우리는 손실함수(Loss function)의 개념을 알아야합니다.  

#### 손실함수(Loss function)
_ _ _

위에서 설명한 방법으로 가중치와 편향의 초기값을 정한 후에 해당 초기값을 이용한 모형으로 새로 입력된 데이터가 어떤 클래스에 속하는지 예측을 수행했다고 가정해봅시다. 예를들어 손으로 쓴 숫자 이미지 데이터를 0 ~ 9까지 중 하나로 분류하는 모형을 만들었는데 다음과 같은 데이터 9로 분류하지 않고 5로 분류했다면 신경망 모형의 정확도가 낮은 것으로 판단할 수 있습니다.  

![](assets/images/mnistnine.JPG)

신경망에서는 이처럼 잘못예측한 것에 대해 손실이 있다고 간주합니다. 9를 5로 분류했다면 손실이 있는 것이고, 9를 9로 분류했다면 손실이 없는 것입니다. 손실함수로 주로 사용되는 함수는 다음과 같습니다.  

1. 평균 제곱 오차(MSE; Mean Squared Error)
2. 교차 엔트로피 오차(CEE; Cross Entropy Error)  

만약 다음과 같은 예측치와 레이블이 주어졌다고 해봅시다. y는 신경망이 예측한 값이고, t는 실제 레이블입니다. 즉 이 경우에 신경망은 3일 확률이 0.6으로 가장 높다는 결과를 냈고, 실제로 정답은 2입니다.  

y = (0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0)  
t = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)  

예측결과는 틀렸습니다. 그렇다면 손실이 있는 것입니다. 손실함수는 예측이 틀린 정도가 강할수록 값이 커지도록 설계되어 있습니다. MSE와 CEE의 수식은 다음과 같습니다.  

![](assets/images/msecee.JPG)

'밑바닥부터 시작하는 딥러닝'에서는 정확도 대신 손실함수값을 지표로 삼는 이유를 미분을 위해서라고 설명하고 있습니다. 손실함수 값은 연속적인 값이 나오기 때문에 gradient descent를 통해 최소가 되는 방향으로 매개변수를 갱신해갈 수 있지만, 정확도는 'N 장중에 k 장을 맞췄다' 처럼 정수값 k에 의해서 결정되기 때문에 연속적인 값이 아닙니다. 그래서 미분을 사용해 아주 조금씩 매개변수를 갱신하면서 최적값을 찾아가는 gradient descent의 적용이 힘들기 때문에 손실함수를 정의해서 지표로 삼아야 한다는 것입니다. 책에서의 설명이 아주 와닿지는 않지만 저는 그렇게 이해했습니다.

#### 미니배치
_ _ _

그러면 신경망의 설계가 끝나고 초기값도 정했으며, 손실함수도 설정했다고 해봅시다. 기계학습 문제는 훈련 데이터(training set)를 이용하여 학습합니다. 더 구체적으로 말하면 훈련 데이터에 대한 손실함수 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아냅니다. 이렇게 하려면 모든 훈련 데이터를 대상으로 손실함수 값을 구해야합니다. 즉, 훈련 데이터가 100개 있으면 그로부터 계산한 100개의 손실함수 값들의 합을 지표로 삼는 것입니다. 그러나 데이터의 크기가 커지면 훈련 데이터로 설정한 모든 데이터에 대해서 손실함수 값을 구하는데 시간이 오래걸립니다. 때문에 미니배치로 특정 숫자를 정해놓고 미니배치의 크기만큼 훈련 데이터를 랜덤 샘플링해서 해당 데이터에 대해서 학습을 진행하는 것입니다.

예를 들어 70000개의 데이터가 있을때 60000개를 훈련데이터로 정했고 미니배치 크기를 100으로 정했다면 60000개 중 100개를 무작위로 뽑아 100개에 대한 손실함수를 계산하고, 계산된 손실함수 값에 대해서 매개변수의 최적화를 수행하게 되는 것입니다.


#### 최적화(optimization - gradient descent)
_ _ _

그렇다면 손실함수의 값이 최소가 되는 매개변수(가중치와 편향)는 어떻게 찾을까요? 앞서 잠시 이야기했듯 미분을 이용해야 합니다. 일반적으로 신경망은 수많은 가중치와 편향 값들에 의해 예측된 값과 실제 값을 기반으로 손실함수를 구하도록 설계되어 있습니다. 그래서 수학적으로 보면 매개변수 값들을 손실함수가 작아지는 방향으로 갱신하는 문제는 각각의 매개변수 값들에 대한 편미분을 구하고, 거기에 아주 작은 값을 곱해서 갱신 값을 만들어 원래의 매개변수에 더해주는 것으로 해결할 수 있습니다. 수식으로 보면 다음과 같습니다.  

![](assets/images/gd.JPG)

빠른 이해를 위해 2차원에서의 gradient descent를 그림으로 보면 다음과 같습니다.  

![](assets/images/gradientdescent.JPG)

이를 이용해서 학습을 수행하는 가장 간단한 방법은 미니배치크기만큼 매개변수 값을 갱신해주는 작업을 여러 번 하는 것입니다. 예를 들어 60000개의 훈련 데이터가 있을때 미니배치 크기를 100으로 설정했다면 gradient descent를 100번씩 600번 수행하는 것입니다. 그럼 총 60000번 수행하는 것이 되겠죠. 그냥 60000번 수행하는 것과 뭐가 다른가 의문이 들 수 있지만 미니배치는 반복문을 돌리는 개념이 아닙니다. 미니배치는 행렬연산으로 수행되기 때문에 100번을 돈다기보다는 100개를 한 번에 연산하는 것으로 이해하면 됩니다. 100개의 미니배치 연산을 600번 수행하는 이유는 그래야 60000개의 훈련 데이터를 모두 소진했다고 보기 때문입니다. 이 600번을 에폭(epoch)이라는 단위로 부릅니다.

그러나 이렇게 연산마다 grdient descent를 수행하는 것은 효율적이지 않습니다. 게다가 1에폭만 학습을 수행하는 것도 아닙니다. 만약 위의 경우처럼 훈련 데이터가 60000개, 미니배치의 크기가 100라면 1에폭은 600번이지만 1에폭은 주로 정확도를 계산하는 주기를 위해 사용할 뿐 반복횟수는 따로 정해줍니다. 반복횟수는 iteration이라고 부르는데, 예컨대 10000번을 iteration으로 설정했다면 100개씩 10000번 매개변수값 갱신의 반복을 수행하는 동안 600번째마다 정확도를 계산합니다. 이것이 역전파를 사용하지 않았을때의 신경망 학습 알고리즘입니다.

#### Two Layer Net
_ _ _

역전파를 살펴보기 전에 신경망이 어떻게 동작하는지 확실히 파악하는 기회를 갖는게 좋겠다는 생각이 들었습니다. 신경망 알고리즘의 그 유명한 오류역전파가 도대체 어떤 점이 좋은지를 이해하기 위해서는 역전파가 적용되기 전의 신경망 작동 원리를 이해하는 것이 중요하다고 생각했기 때문입니다. 파이썬으로 구현된 2층 신경망을 동작시키는 소스는 다음과 같습니다.


train_neuralnet.py
<pre><code>
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# TwoLayerNet 객체 생성(입력층784 - 은닉층50 - 출력층10)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  				# 반복 횟수를 적절히 설정한다.(10000번)
train_size = x_train.shape[0]   # 훈련데이터셋 개수 (60000개)
batch_size = 100   				# 미니배치 크기 (100개)
learning_rate = 0.1				# 학습률 (0.1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수 (60000/100 = 600번이면 1에폭이고, 데이터를 한 번 다 순환한것으로 간주한다)
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
</code></pre>


TwoLayerNet.py

<pre><code>
# coding: utf-8 
import sys, os 
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정 
from common.functions import * 
from common.gradient import numerical_gradient 

 
class TwoLayerNet: 

 
     def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01): 
         # 가중치 초기화 
         self.params = {} 
           self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) 
         self.params['b1'] = np.zeros(hidden_size) 
         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
         self.params['b2'] = np.zeros(output_size) 
 
 
     def predict(self, x): 
         W1, W2 = self.params['W1'], self.params['W2'] 
         b1, b2 = self.params['b1'], self.params['b2'] 
      
         a1 = np.dot(x, W1) + b1 
         z1 = sigmoid(a1) 
         a2 = np.dot(z1, W2) + b2 
         y = softmax(a2) 
          
         return y 
          
     # x : 입력 데이터, t : 정답 레이블 
     def loss(self, x, t): 
         y = self.predict(x) 
          
         return cross_entropy_error(y, t) 
      
     def accuracy(self, x, t): 
         y = self.predict(x) 
         y = np.argmax(y, axis=1) 
         t = np.argmax(t, axis=1) 
          
         accuracy = np.sum(y == t) / float(x.shape[0]) 
         return accuracy 
          
     # x : 입력 데이터, t : 정답 레이블 
     def numerical_gradient(self, x, t): 
         loss_W = lambda W: self.loss(x, t) 
          
         grads = {} 
         grads['W1'] = numerical_gradient(loss_W, self.params['W1']) 
         grads['b1'] = numerical_gradient(loss_W, self.params['b1']) 
         grads['W2'] = numerical_gradient(loss_W, self.params['W2']) 
         grads['b2'] = numerical_gradient(loss_W, self.params['b2']) 
          
         return grads 

</code></pre>

위에서 보여준 2층 신경망 소스가 가지고 있는 메서드는 다음과 같습니다.


TwoLayerNet.py  


| 메서드명 | 역할 |
|--------|--------|
|   init     |  가중치 초기화      |
|   predict     |    예측수행    |
|   loss     |  손실함수 값 계산      |
|   accuracy     |   정확도 계산     |
|   numerical_gradient     |   역전파 적용 전 기울기 값 계산    |
|   gradient     |   역전파 적용 후 기울기 값 계산    |

역전파를 적용하기 전에는 기울기를 어떻게 구하는지 한 번 봅시다. 2층 신경망이고, 미니배치의 크기는 100입니다.

train_neuralnet.py소스를 보면 TwoLayerNet의 numerical_gradient 메서드에 x_batch와 t_batch를 던져줍니다. 각각은 60000개의 샘플에서 무작위로 선택된 100개의 features와 label입니다. 그렇기 때문에 작동원리를 알기 위해서는 TwoLayerNet으로 와서 numerical_gradient 메서드를 다시 봐야합니다. 여기서는 일단 x_batch를 통해 예측된 값과 t_batch값의 차이를 통해 loss function을 재정의합니다. 이 loss function은 predict와 cross_entropy_error의 계산을 포함하고 있습니다. 그리고 나서 각각의 가중치 값들에 대한 기울기를 구하는데, 여기서 헷갈릴 수 있는 점은 numerical_gradient 안에 있는 numerical_gradient 메서드는 재귀호출이 아니라 common 함수에 있는 numerical_gradient 라는 점입니다. 해당 함수의 소스는 다음과 같습니다.


<pre><code>
def numerical_gradient(f, x): 
     h = 1e-4 # 0.0001 
     grad = np.zeros_like(x) 
      
     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) 
     while not it.finished: 
         idx = it.multi_index 	# 다차원 배열의 인덱스를 튜플로 반환 
         tmp_val = x[idx] 
         x[idx] = float(tmp_val) + h 
         fxh1 = f(x) # f(x+h) 
          
         x[idx] = tmp_val - h  
         fxh2 = f(x) # f(x-h) 
         grad[idx] = (fxh1 - fxh2) / (2*h) 
          
         x[idx] = tmp_val # 값 복원 
         it.iternext()    
          
     return grad 
</code></pre>

그리고 크로스 엔트로피 오차를 구하는 함수는 다음과 같습니다.

<pre><code>
def cross_entropy_error(y, t): 
     if y.ndim == 1: 
         t = t.reshape(1, t.size) 
         y = y.reshape(1, y.size) 
          
     # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환 
     if t.size == y.size: 
         t = t.argmax(axis=1) 
               
     batch_size = y.shape[0] 
     return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size 
</code></pre>

numerical_gradient에서는 x로 입력된 행렬의 각 요소를 한 번씩 순회하면서 함수에 대한 편미분 값(기울기)를 구합니다. 때문에 x의 요소 개수만큼 반복문을 돌게됩니다. 여기서는 각 가중치 행렬(W1, B1, W2, B2)이 순차로 들어왔기 때문에 가중치행렬의 요소 개수만큼(W1의 경우에는 (784,50) - 39200) 루프를 돌게 되겠지요. 이렇게 루프를 돌면 기울기 값이 계산되고, train_neuralnet.py에서는 iteration을 10000번으로 정해줬기 때문에 39200번 외에도 3개의 루프에 대한 루프를 10000번 돌게 되는 것입니다. 실제로 소스를 돌려보면 한 번 학습하는데 걸리는 시간이 매우 깁니다. 그래서 우리는 이것보다 더 효율적인 방법으로 기울기를 계산할 것입니다. 오류역전파를 적용한다는 것은 그런 의미입니다.

#### 역전파(backward propagation)
_ _ _

'밑바닥부터 시작하는 딥러닝'에서는 오차역전파(backpropagation)를 계산그래프로 설명하고 있습니다. 계산그래프를 이용하는 이점은 첫째로 국소적 계산이 가능하다는 것이고, 둘째로 중간계산결과를 모두 보관할 수 있다는 것입니다. 우리가 구하고자 하는 것은 무엇인가요? 가중치의 변화에 따른 손실함수값의 변화입니다. 가중치값의 순간변화량에 대한 손실함수의 순간변화량, 즉 미분값이죠. 계산그래프를 통해서 보면, 우리가 국소적 미분의 전달을 통해 전체 미분을 구할 수 있다는 사실을 알 수 있습니다. 계산그래프를 하나하나 설명하지는 않겠습니다. 이 포스팅에서는 역전파가 신경망 안에서 어떻게 구현되었는지에만 초점을 맞추겠지만, 정말 기초적인 원리부터 알고싶다면 '밑바닥부터 시작하는 딥러닝'의 오차역전파법 챕터를 참고하시면 좋을 것입니다. 역전파의 핵심은 미분값을 순전파의 반대방향으로 전달하는 것이기 때문에 곱셈노드의 역전파와 함수노드의 역전파 그림을 잠시 보고 넘어가겠습니다.

![](assets/images/mulbackward.JPG)

![](assets/images/funcbackward.JPG)

우리가 하는 계산들의 종류는 사칙연산이라고는 하지만 뺄셈은 덧셈으로, 나눗셈은 곱셈으로 모두 표현이 가능합니다. 여기서는 덧셈과 곱셈의 역전파를 구현함으로써 모든 연산에 대한 역전파를 실행할 수 있도록 합니다. 신경망에서 우리가 어떤 연산을 수행했었는지 기억하실겁니다. 입력값들과 가중치의 행렬곱(내적) 연산을 수행한 이후에 편향을 더해서 a값을 도출해내고, a값을 정의역으로 하는 ReLu function을 사용하여 y값을 계산하고 이를 다음 은닉층의 입력값으로 보내주었습니다. 그럼 하나의 층에서는 곱셈+덧셈+함수연산 이 일어난다는 사실이 이해가 가실겁니다. 이 함수연산은 또한 조건에 따른 덧셈과 곱셈연산이겠지요. 그렇게때문에 역전파를 구현하기 위해서는 각각의 계층에 대한 모듈을 구현해야합니다. 이 내용을 표로 보면 다음과 같습니다.  


| 연산 | 계층(모듈) |
|--------|--------|
| 행렬곱 + 편향 | Affine |
| 활성함수  | ReLU 또는 Sigmoid |
| 출력함수  | Softmax with loss |  

이해를 위해서 Affine 모듈의 연산과정에 대한 그림을 보면 다음과 같습니다.

![](assets/images/affinebackward.JPG)

각 모듈에서는 forward(순전파)와 backward(역전파)를 모두 구현하게 됩니다. 그리하여 입력 데이터가 각 계층에서 순전파로 흘러감과 동시에 역전파로 기울기를 구해 값을 보내줄 수 있는 것입니다. 출력함수 모듈의 이름이 softmax with loss인 것은 softmax함수로 출력값(예측값)을 구한 이후에 정답레이블과 비교하여 교차엔트로피오차를 구하는 과정까지 포함되어있기 때문입니다. 역전파를 적용하기 전과 비교하면 어떨까요? 사실 저는 이 부분이 이해가 잘 가지 않아 한참을 생각해야 했습니다. 역전파가 왜 빠를까? 두 방법을 비교하는 직관적인 그림이 없어서 그랬던 것 같기도 합니다. 일단 역전파가 적용된 2층 신경망의 소스를 보며 이야기를 이어가겠습니다.


train_neuralnet.py
<pre><code>
# coding: utf-8 
import sys, os 
sys.path.append(os.pardir) 


import numpy as np 
from dataset.mnist import load_mnist 
from two_layer_net import TwoLayerNet 

 
# 데이터 읽기 
 (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) 
 
 
 network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10) 
 
 
 iters_num = 10000 
 train_size = x_train.shape[0] 
 batch_size = 100 
 learning_rate = 0.1 
 
 
 train_loss_list = [] 
 train_acc_list = [] 
 test_acc_list = [] 
 
 
 iter_per_epoch = max(train_size / batch_size, 1) 
 
 
 for i in range(iters_num): 
     batch_mask = np.random.choice(train_size, batch_size) 
     x_batch = x_train[batch_mask] 
     t_batch = t_train[batch_mask] 
      
     # 기울기 계산 
     #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식 
     grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다) 
      
     # 갱신 
     for key in ('W1', 'b1', 'W2', 'b2'): 
         network.params[key] -= learning_rate * grad[key] 
      
     loss = network.loss(x_batch, t_batch) 
     train_loss_list.append(loss) 
      
     if i % iter_per_epoch == 0: 
         train_acc = network.accuracy(x_train, t_train) 
         test_acc = network.accuracy(x_test, t_test) 
         train_acc_list.append(train_acc) 
         test_acc_list.append(test_acc) 
         print(train_acc, test_acc) 
         
</code></pre>

train_neuralnet.py은 앞에서 구현했던 것과 차이가 없습니다. 다만 TwoLayerNet에서 numerical_gradient메서드 대신에 gradient에서드를 사용하는 것만 바뀌었습니다. 그렇다면 TwoLayerNet.py는 어떨까요? 여기에는 많은 변화가 있습니다.


TwoLayerNet.py
<pre><code>
# coding: utf-8 
import sys, os 
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정 
import numpy as np 
from common.layers import * 
from common.gradient import numerical_gradient 
from collections import OrderedDict 
 
 class TwoLayerNet: 
 
 
     def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01): 
         # 가중치 초기화 
         self.params = {} 
         self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) 
         self.params['b1'] = np.zeros(hidden_size) 
         self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  
         self.params['b2'] = np.zeros(output_size) 
 
         # 계층 생성 
         self.layers = OrderedDict() 
         self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1']) 
         self.layers['Relu1'] = Relu() 
         self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2']) 
 
         self.lastLayer = SoftmaxWithLoss() 
          
     def predict(self, x): 
         for layer in self.layers.values(): 
             x = layer.forward(x) 
          
         return x 
          
     # x : 입력 데이터, t : 정답 레이블 
     def loss(self, x, t): 
         y = self.predict(x) 
         return self.lastLayer.forward(y, t) 
      
     def accuracy(self, x, t): 
         y = self.predict(x) 
         y = np.argmax(y, axis=1) 
         if t.ndim != 1 : t = np.argmax(t, axis=1) 
          
         accuracy = np.sum(y == t) / float(x.shape[0]) 
         return accuracy 
          
     # x : 입력 데이터, t : 정답 레이블 
     def gradient(self, x, t): 
         # forward 
         self.loss(x, t) 
 
         # backward 
         dout = 1 
         dout = self.lastLayer.backward(dout) 
          
         layers = list(self.layers.values()) 
         layers.reverse() 
         for layer in layers: 
             dout = layer.backward(dout) 
 
 
         # 결과 저장 
         grads = {} 
         grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db 
         grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db 
 
         return grads 

</code></pre>


우션 계층 생성단계가 추가되었습니다. 각 계층에 대한 계산 모듈을 따로 구현했기 때문에 이러한 모듈을 조립해서 신경망을 구성하는 일이 가능해진 것입니다. 그리고 numerical_gradient의 자리를 gradient가 대체했습니다. 미분값으로 처음에 1을 전달해주면 각 계층들에 대해 루프를 돌면서 backward를 호출하여 국소적 미분값의 전달을 통해 최초에 입력된 변수에 대한 미분값(기울기)을 구하게 됩니다. 각 계층의 구현 소스는 다음과 같습니다.

common/Layer.py 의 일부
<pre><code>
# coding: utf-8 
import numpy as np 
from common.functions import * 
from common.util import im2col, col2im 

class Relu: 
    def __init__(self): 
        self.mask = None 


     def forward(self, x): 
         self.mask = (x <= 0) 
         out = x.copy() 
         out[self.mask] = 0 

         return out 
 
     def backward(self, dout): 
         dout[self.mask] = 0 
         dx = dout 
 
         return dx 

 class Sigmoid: 
     def __init__(self): 
         self.out = None 
 
 
     def forward(self, x): 
         out = sigmoid(x) 
         self.out = out 
         return out 
 
     def backward(self, dout): 
         dx = dout * (1.0 - self.out) * self.out 
 
         return dx 
 
 class Affine: 
     def __init__(self, W, b): 
         self.W = W 
         self.b = b 
          
         self.x = None 
         self.original_x_shape = None 
         # 가중치와 편향 매개변수의 미분 
         self.dW = None 
         self.db = None 
 
     def forward(self, x): 
         # 텐서 대응 
         self.original_x_shape = x.shape 
         x = x.reshape(x.shape[0], -1) 
         self.x = x 
 

         out = np.dot(self.x, self.W) + self.b 
 
         return out 
 
     def backward(self, dout): 
         dx = np.dot(dout, self.W.T) 
         self.dW = np.dot(self.x.T, dout) 
         self.db = np.sum(dout, axis=0) 
          
         dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응) 
         return dx 

 class SoftmaxWithLoss: 
     def __init__(self): 
         self.loss = None # 손실함수 
         self.y = None    # softmax의 출력 
         self.t = None    # 정답 레이블(원-핫 인코딩 형태) 
          
     def forward(self, x, t): 
         self.t = t 
         self.y = softmax(x) 
         self.loss = cross_entropy_error(self.y, self.t) 
          
         return self.loss 
 
     def backward(self, dout=1): 
         batch_size = self.t.shape[0] 
         if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때 
             dx = (self.y - self.t) / batch_size 
         else: 
             dx = self.y.copy() 
             dx[np.arange(batch_size), self.t] -= 1 
             dx = dx / batch_size 
          
         return dx 
</code></pre>

역전파를 사용하기 전에는 초기화된 가중치를 이용해서 배치 데이터로 일단 예측을 수행한 이후에, 해당 가중치 행렬을 하나씩 집어넣어 각 요소에 대해서 루프를 돌면서 수치미분 메서드를 통해 편미분 값을 구했습니다. 그러나 역전파를 적용하고 나서는 각 계층의 역전파를 통해 행렬값이 흐르면서 기울기값을 계산해서 리턴해줍니다. 실제로 각 소스코드를 실행시켜보면 역전파를 적용한 소스의 학습이 훨씬 빠르다는 것을 알 수 있습니다.

이와같이 신경망에서는 역전파를 이용하여 기울기를 효율적으로 구합니다. 그리고 계층단위의 모듈을 구현하는 것은 기울기의 효율적 계산뿐 만아니라 원하는 설계대로 각 모듈들을 조립하여 신경망을 구현할 수 있게 해줍니다. 이번 포스팅에서는 2층 신경망만을 다뤘지만, 다음 포스팅에서는 CNN과 RNN, 그리고 층을 깊게한 deep neural network에 대해서 이야기해보겠습니다.

http://blog.naver.com/yhbest12/220973959969

