from flask import Flask, render_template, request, jsonify, session
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)
app.secret_key = 'capston'
# 모델 불러오기

model = models.resnet18(weights=None)

# 기존의 출력 레이어를 새로운 클래스 수에 맞게 수정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 2는 새 클래스 수에 맞게 조정(피부암인지 아닌지),  해당 데이터셋 클래스에 맞게 하기
model.load_state_dict(torch.load('resnet18_class10.pth'))  # 이름수정확인
model.eval()

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('result.html', result="Upload an image and click 'Predict'")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트에서 이미지 파일 받기
        uploaded_file = request.files['file']

        # 이미지 전처리 및 예측
        image = Image.open(uploaded_file).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)

        # 결과 전송
        _, predicted_idx = torch.max(output, 1)
        result = f'Predicted class: {predicted_idx.item()}'
        # 세션에 결과 저장
        session['prediction_result'] = predicted_idx.item()
        return jsonify({'result': result})  # JSON 형식으로 결과 반환
    except Exception as e:
        return jsonify({'error': str(e)})  # 오류 메시지를 JSON 형식으로 반환

@app.route('/get_result')
def get_result():
    result = session.get('prediction_result', None)
    # 여기서 원하는 결과를 생성
    if result is not None:
        dataset_labels = ['Akne', 'Benign', 'Ekzama', 'Enfeksiyonel', 'Malign']  # 데이터셋 클래스 레이블을 예시로 표시  , 클래스 수정하기
    #class_index_to_name = {i: label for i, label in enumerate(dataset_labels)}
        predicted_class_name = dataset_labels[result]
        return jsonify(f'This is the  {predicted_class_name}')
    else:
        return jsonify({'error': 'None'})

if __name__ == '__main__':
    app.run(debug=True)