import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt


model = models.resnet18(weights=None)

# 기존의 출력 레이어를 새로운 클래스 수에 맞게 수정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 2는 새 클래스 수에 맞게 조정(피부암인지 아닌지),  해당 데이터셋 클래스에 맞게 하기
model.load_state_dict(torch.load('resnet18_class10.pth'))  # 이름수정확인

# 모델을 평가 모드로 설정
model.eval()

input_image = Image.open('C:/캡디/dataset/kaggle/val/Ekzama/0_13.jpg')  # 경로지정
result_txt = 'C:/Users/user/OneDrive - 동의대학교/바탕 화면/3학년/2/웹프/캡디/hi.txt'
# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # 모델이 요구하는 배치 차원 추가

# 이미지 예측
with torch.no_grad():
    output = model(input_batch)

# 예측 결과 가져오기
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 확률이 가장 높은 클래스 인덱스 얻기
predicted_class = torch.argmax(probabilities).item()

# 데이터셋 클래스 레이블
dataset_labels = ['Akne','Benign', 'Ekzama', 'Enfeksiyonel', 'Malign']  # 데이터셋 클래스 레이블을 예시로 표시  , 클래스 수정하기

# 클래스 인덱스를 클래스 이름으로 매핑하는 딕셔너리 생성
class_index_to_name = {i: label for i, label in enumerate(dataset_labels)}

predicted_class_name = class_index_to_name[predicted_class]

# 이미지 및 예측 텍스트 출력
plt.imshow(input_image)
plt.title(f'this is {predicted_class_name} ')
plt.axis('off')
plt.show()

with open(result_txt, 'w') as result_file:
    result_file.write(predicted_class_name)
