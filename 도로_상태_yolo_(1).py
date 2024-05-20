import subprocess
import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from PIL import Image
import ultralytics
from ultralytics import YOLO
import yaml

# pip 설치
subprocess.run(['pip', 'install', 'kaggle', '--upgrade'], check=True)
subprocess.run(['pip', 'install', 'wandb==0.15.0'], check=True)
subprocess.run(['pip', 'install', 'ultralytics'], check=True)

# Kaggle API 설정
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# 데이터셋 다운로드 및 압축 해제
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mersico/road-damage-tracking-dataset-rdtd-v10'], check=True)
subprocess.run(['unzip', 'road-damage-tracking-dataset-rdtd-v10.zip'], check=True)

# wandb 로그인
subprocess.run(['wandb', 'login', '84067f184707d2c92577dc4a6d85060acd980f35'], check=True)

# 데이터셋 디렉토리 생성
os.makedirs('datasets/train', exist_ok=True)
os.makedirs('datasets/valid', exist_ok=True)
os.makedirs('datasets/test', exist_ok=True)

train_path = 'datasets/train/'
valid_path = 'datasets/valid/'
test_path = 'datasets/test/'

ano_paths = []
i = 0
for dirname, _, filenames in os.walk('/content/train-pure/train-pure/labels'):
    for filename in filenames:
        if filename.endswith('.txt') and i < 400:
            ano_paths.append(os.path.join(dirname, filename))
            i += 1

n = len(ano_paths)
print(n)
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
print(len(os.listdir(train_path)))

for i in valid_i:
    ano_path = ano_paths[i]
    img_path = os.path.join('/content/train-pure/train-pure/images', ano_path.split('/')[-1][0:-4] + '.jpg')
    try:
        shutil.copy(ano_path, valid_path)
        shutil.copy(img_path, valid_path)
    except:
        continue
print(len(os.listdir(valid_path)))

for i in test_i:
    ano_path = ano_paths[i]
    img_path = os.path.join('/content/train-pure/train-pure/images', ano_path.split('/')[-1][0:-4] + '.jpg')
    try:
        shutil.copy(ano_path, test_path)
        shutil.copy(img_path, test_path)
    except:
        continue
print(len(os.listdir(test_path)))

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
    plt.show()

best_path0 = '/content/runs/detect/train3/weights/best.pt'
source0 = 'datasets/test'

ppaths = []
for dirname, _, filenames in os.walk(source0):
    for filename in filenames:
        if filename.endswith('.jpg'):
            ppaths.append(os.path.join(dirname, filename))
ppaths = sorted(ppaths)
print(ppaths[0])
print(len(ppaths))

model2 = YOLO(best_path0)

subprocess.run(['yolo', 'task=detect', 'mode=predict', 'model=' + best_path0, 'conf=0.1', 'source=' + source0], check=True)

results = model2.predict(source0, conf=0.1)
print(len(results))

print((results[0].boxes.data))

PBOX = pd.DataFrame(columns=range(6))
for i in range(len(results)):
    arri = pd.DataFrame(results[i].boxes.data.cpu().numpy()).astype(float)
    path = ppaths[i]
    file = path.split('/')[-1]
    arri = arri.assign(file=file)
    arri = arri.assign(i=i)
    PBOX = pd.concat([PBOX, arri], axis=0)
PBOX.columns = ['x', 'y', 'x2', 'y2', 'confidence', 'class', 'file', 'i']
display(PBOX)

PBOX['class'] = PBOX['class'].apply(lambda x: class_map[int(x)])
PBOX = PBOX.reset_index(drop=True)
display(PBOX)
display(PBOX['class'].value_counts())

def draw_box2(n0):
    ipath = ppaths[n0]
    image = cv2.imread(ipath)
    H, W = image.shape[0], image.shape[1]
    file = ipath.split('/')[-1]

    if PBOX[PBOX['file'] == file] is not None:
        box = PBOX[PBOX['file'] == file]
        box = box.reset_index(drop=True)

        for i in range(len(box)):
            label = box.loc[i, 'class']
            x = int(box.loc[i, 'x'])
            y = int(box.loc[i, 'y'])
            x2 = int(box.loc[i, 'x2'])
            y2 = int(box.loc[i, 'y2'])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)  # green

    return image

def create_animation(ims):
    fig = plt.figure(figsize=(12, 8))
    im = plt.imshow(cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB))
    text = plt.text(0.05, 0.05, f'Slide {0}', transform=fig.transFigure, fontsize=14, color='blue')
    plt.axis('off')
    plt.close()

    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
        text.set_text(f'Slide {i}')
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000)

images2 = []
for i in tqdm(range(len(ppaths))):
    images2.append(draw_box2(i))

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 30  # 원하는 크기 (MB)로 설정

create_animation(images2)
