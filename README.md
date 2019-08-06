# faceRecognition 人脸识别

## 通过TensorFlow训练的模型完成人脸识别与相似度对比

### 1.人脸检测:可以用MTCNN模型或Android自带人脸检测完成(MTCNN速度稍慢但精确度更好)

### 2.人脸特征提取:图片经过预处理后通过TensorFlow模型得到特征数组

### 3.相似度对比:先计算两个特征数组的欧式距离(距离越小可以认为两个特征相似度越高)，通过多个样本数据拟合出距离与相似度的转换公式，然后得到相似度


## 截图

![杨超越](https://github.com/wentaibao/faceRecognition/blob/master/Screenshots/Screenshot_1.png?raw=true)

![篮球](https://github.com/wentaibao/faceRecognition/blob/master/Screenshots/Screenshot_2.png?raw=true)
