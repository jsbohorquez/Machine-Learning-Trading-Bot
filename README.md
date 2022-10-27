# Challenge-14
# Machine Learning Trading Bot

## Overview
The goal of this project is to improve an existing trading algorithm using machine learning that can be used to assess and adapt to new data. By setting up a performance baseline, it is then tuned, and finally evaluated using a new ML classifier. Finally, a performance report can then be produced.


## Technologies Used
The libraries and dependencies used in this program are listed in the first cell of the program which are: 
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

## Program Layout
This program is divided into the following parts:

**Part 1** - Baseline Performance
            Code and data was given and used to generate long and short window simple moving averages (SMA) were produced. This was then applied to the SVC classifier
            model, by fitting the training data to make predictions using the testing data.
            The image below shows the cumulative return plot of the Actual Returns vs Strategy Returns as well as its corresponding classification report:
![Screenshot (25)](https://user-images.githubusercontent.com/101238359/184468239-78465dfe-2fc2-43bb-b826-7dce78f46b78.png)
![Screenshot (26)](https://user-images.githubusercontent.com/101238359/184468240-48b75532-4373-41b3-aefa-5b18513c22a3.png)      
        
            
**Part 2** - Applied New Machine Learning Classifier (Ada Boost)
            For this section, the aforementioned classifier was used to create an alternate model using the data provided for 'Part 1'. 
            Below the Actual Returns vs Strategy Returns Plot is shown for this classifier (Ada Boost):

![Screenshot (27)](https://user-images.githubusercontent.com/101238359/184468242-10e1cb83-351e-4bd5-bb92-170d6758a75d.png)
![Screenshot (28)](https://user-images.githubusercontent.com/101238359/184468244-85f7980c-0536-4550-9494-2148554720a0.png)



## Author
Juan Bohorquez

Reference and guides - Trilogy Education Services.
