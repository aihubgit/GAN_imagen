## 모델 description 

GAN(Imagen)은 두개의 모델이 있습니다. 텍스트(언어) 이해를 위한 Transformer기반의 언어 모델과 고품질의 이미지를 생성할 수 있는 확산 모델로 이루어져 있습니다.

사용한 언어모델은 텍스트 만으로 사전학습된 'T5' 입니다. 

사용한 확산 모델은 'Efficient U-Net' 입니다.

## 모델 아키텍쳐

<p align='center'>
<img width="673" alt="imagen" src="https://user-images.githubusercontent.com/120080865/210080451-a3e45805-113d-49b3-b81d-05a14f6d7519.png">
<p align='center'>
GAN(imagen) 모델 구조

 
## input
 - 입력은 [0, 255] 범위의 값을 가진 16bit, 3채널 RGB 이미지 입니다.
 - 모양 : (B, H, W, 3)
 
## output
 - 출력은 [0, 255] 범위의 값을 가진 3채널 RGB 이미지 입니다.
 - 모양 : (B, H, W, 3)
 
## task
 - 텍스트 기반 이미지 생성

## training dataset
 심볼[로고] 이미지(50k)
 - 훈련에 100% 사용(이미지 50만개)
 
 
## training 요소들
 
 - epoch(학습량) : iter(200000)
 - input_image_size(입력 이미지 크기) : 1024
 - output_image_size(출력 이미지 크기) : 256
 - batch_size(한번 학습에 입력되는 데이터 수) : 4
 - learning_rate(학습률) : 1e-4
 

## evaluation metric
 - 성능 Plot 이나 Accuracy 등 기본적인 모델 성능 지표 및 수치, 유효성 검증 목표 지표와 달성 수치 기재
