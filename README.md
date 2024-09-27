# Rep_Mask_RCNN
 在UIIS数据集上训练Mask RCNN
## 环境配置：
* Python3.6以上
* Pytorch1.10或以上
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`)
* 使用Ubuntu系统(不建议Windows)
* 详细环境配置见`requirements.txt`
### Datasets
请在项目中创建一个data文件夹，并按照以下格式将UIIS数据集放入其中。

    data
      ├── UDW
      |   ├── annotations
      │   │   │   ├── train.json
      │   │   │   ├── val.json
      │   ├── train
      │   │   ├── L_1.jpg
      │   │   ├── ......
      │   ├── ......

## 训练方法
* 确保提前准备好数据集
* 确保设置好`--num-classes`和`--data-path`
* 直接使用train.py进行训练
  
## 注意事项
在使用训练脚本时，注意要将`--data-path`设置为自己存放数据集的**根目录**：
