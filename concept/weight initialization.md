gradient vanishing 문제 발생

→ activation function을 바꾼다.

→ 가중치 초기화 방법

가중치 초기화는 왜?

1. zero initialization

    동일한 값으로 초기화 하는 것은 하나의 레이어에 하나의 노드가 있는 것과 동일함.

    backpropagation과정을 생각하면 모두 동일하게 가중치가 학습되어 여러개의 노드가 있는 것이 의미가 없어짐.

2. random initialization

    그래서 랜덤으로 초기화를 시켜보자

    - 일단 모든 가중치가 평균이 0이고 표준편차가 1인 정규분포를 이루고 있다.

        → 출력값이 0과 1에 치우치는 현상이 발생함

        sigmoid함수에서 gradient가 0에 가까우며 gradient vanishing문제가 발생함

        weight가 0에서 멀수록 = 표준편차가 클수록 sigmoid function을 사용하면 0과 1에 치우치는 현상이 발생함.

    - 표준편차가 0.01인 정규분포

        0.5중심으로 모여있음 → 0과1일때보다 의미있는 gradient를 갖게되어 gradient vanishing현상 완화 

3. RBM →현재는 잘 사용안함 왜냐면 간단한 4,5번 방법이 등장했기때문이지!!

    step

    1. pretraining
    2. fine-tuning
4. xavier initialization
    1. normal initialization (정규분포)
    2. uniform initialization (연속균등할당분포)

    ```python
    torch.nn.init.xavier_uniform_(linear.weight)
    ```

5. he initialization →xavier의 응용
    - n_out term만 없애면 he 초기화지롱
