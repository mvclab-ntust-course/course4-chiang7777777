## 資料集

[Animals Image Dataset](https://universe.roboflow.com/roboflow-100/animals-ij5d2/dataset/2)
* TRAIN SET : 700 Images
* VALID SET : 200 Images
* TEST  SET : 100 Images
* example :  
![image](https://i.imgur.com/ZoWSEV8.jpeg)

## 下載資料集
```python=
from roboflow import Roboflow
rf = Roboflow(api_key="wA7nupKJUAf9fJh7xSu6")
project = rf.workspace("roboflow-100").project("animals-ij5d2")
version = project.version(2)
dataset = version.download("yolov8")
```

## 訓練
```python=
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='animals-2\data.yaml', epochs=100, imgsz=640)
```
Loss curve :
![image](https://i.imgur.com/7HgaCHi.png)



## 預測
```python=
model = YOLO("./runs/detect/train2/weights/best.pt")
result = model.predict(
    source="./animals-2/test/images",
    mode="predict",
    save=True,
)
```
result : 
![image](https://i.imgur.com/PxzT6CN.jpeg)
