# buaa2025-sjwj
2025秋buaa数据挖掘导论大作业


## 项目结构

*   `image_clustering.ipynb`: 任务一
*   `weather_forecasting.ipynb`: 任务三
*   `thyroid_disease_detection.ipynb`: 任务四
*   `DM_2025_Dataset/`: 数据集
*   `Clustered_Results/`: 任务一的图像聚类结果输出
*   `requirements.txt`: 项目依赖列表

## 环境要求

请确保您的机器上安装了 Python 3.8 或更高版本。


### 安装依赖

在项目根目录下，打开终端并运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```


### 数据集准备

请确保 `DM_2025_Dataset` 文件夹位于项目根目录下，且结构如下：

```text
DM_2025_Dataset/
    weather.csv              
    Cluster/                 
    Image_Anomaly_Detection/
    thyroid/                  
task2/
...
```
### 运行任务

任务一： 运行 `image_clustering.ipynb` 

任务二： 运行 `task2/src/train_eval.py`，支持命令行参数配置。

在该任务中，我们针对不同类别得出了最佳配置，请使用如下命令进行评估：

*   **Hazelnut (榛子类)**：
    ```bash
    python task2/src/train_eval.py --categories hazelnut --score-mode one-class --image-size 256 --use-pca --pca-dim 128
    ```
*   **Zipper (拉链类)**：
    ```bash
    python task2/src/train_eval.py --categories zipper --score-mode one-class --image-size 384 --use-pca --pca-dim 128
    ```

任务三： 运行 `weather_forecasting.ipynb`

任务四： 运行 `thyroid_disease_detection.ipynb`