# PMCKT
Prompt-Enhanced Multi-Modal Cross-region Knowledge Transfer for Next POI Recommendations.
# Datasets
Our datasets are collected from the following links. https://drive.google.com/drive/folders/1arGcLk9hUrL8XhyUM1iAYPiZZV6ZFH5A?usp=drive_link.  
# Quick Start
1.model_ST.py is the Python code that stores the model.  
2.main_ST.py is the code for training the model, saving the model weight file and obtaining the result.  
3.datasets_ST.py is the code for loading training data. This code can choose whether to load all region data for the first stage training or fine-tune the target region data for the second stage based on the region marker bit. This code can increase the offset for POI numbers in other fields to ensure the uniqueness of POI numbers.  
4.utils.py is the test function code for testing recall and ndcg indicators.  
# Requirements
python 3.8+  
pytorch 1.11.0
