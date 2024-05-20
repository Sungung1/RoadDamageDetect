
import os
import subprocess
import shutil
import random
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from PIL import Image
from ultralytics import YOLO
import yaml

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
DATASET_FOLDER = 'datasets'
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, 'train')
VALID_FOLDER = os.path.join(DATASET_FOLDER, 'valid')
TEST_FOLDER = os.path.join(DATASET_FOLDER, 'test')

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 파일 업로드 및 처리
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # YOLO 모델 실행
        run_yolo(filepath)
        
        return redirect(url_for('results'))
    return redirect(request.url)

# 결과 페이지
@app.route('/results')
def results():
    results_path = os.path.join(RESULT_FOLDER, 'output.jpg')
    return render_template('results.html', results_image=results_path)

def run_yolo(filepath):
    # Kaggle API 설정
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

    # 데이터셋 다운로드 및 압축 해제
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mersico/road-damage-tracking-dataset-rdtd-v10'], check=True)
    subprocess.run(['unzip', 'road-damage-tracking-dataset-rdtd-v10.zip'], check=True)

    # 데이터셋 디렉토리 생성
    train_path = TRAIN_FOLDER
    valid_path = VALID_FOLDER
    test_path = TEST_FOLDER

    ano_paths = []
    i = 0
    for dirname, _, filenames in os.walk('/content/train-pure/train-pure/labels'):
        for filename in filenames:
            if filename.endswith('.txt') and i < 400:
                ano_paths.append(os.path.join(dirname, filename))
                i += 1

    n = len(ano_paths)
    N = list(range(n))
    random.shuffle(N)

    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2

    train_size = int(train_ratio * n)
    valid_size = int(valid_ratio * n)

    train_i = N[:train_size]
    valid_i = N[train_size:train_size + valid_size]
    test_i = N[train_size + valid_size:]

    for i in train_i:
        ano_path = ano_paths[i]
        img_path = os.path.join('/content/train-pure/train-pure/images', ano_path.split('/')[-1][0:-4] + '.jpg')
        try:
            shutil.copy(ano_path, train_path)
            shutil.copy(img_path, train_path)
        except:
            continue

    for i in valid_i:
        ano_path = ano_paths[i]
        img_path = os.path.join('/content/train-pure/train-pure/images', ano_path.split('/')[-1][0:-4] + '.jpg')
        try:
            shutil.copy(ano_path, valid_path)
            shutil.copy(img_path, valid_path)
        except:
            continue

    for i in test_i:
        ano_path = ano_paths[i]
        img_path = os.path.join('/content/train-pure/train-pure/images', ano_path.split('/')[-1][0:-4] + '.jpg')
        try:
            shutil.copy(ano_path, test_path)
            shutil.copy(img_path, test_path)
        except:
            continue

    data_yaml = dict(
        train='datasets/train',
        val='datasets/valid',
        test='datasets/test',
        nc=1,
        names=['damage']
    )

    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

    model = YOLO("yolov8x.pt")

    subprocess.run(['yolo', 'task=detect', 'mode=train', 'data=data.yaml', 'epochs=12', 'imgsz=480'], check=True)

    paths2 = []
    for dirname, _, filenames in os.walk('/content/runs/detect/train3'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                paths2.append(os.path.join(dirname, filename))
    paths2 = sorted(paths2)

    for path in paths2:
        image = Image.open(path)
        image = np.array(image)
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.savefig(os.path.join(RESULT_FOLDER, 'output.jpg'))
        plt.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
