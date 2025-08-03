# FER-with-HOG-and-SVM
Experimental attempt on integrating HOG and SVM for facial expression recognition (FER) problem in Computer Vision


## Feature
- Facial expression classification using **SVM with polynomial kernel**
- Uses **HOG (Histogram of Oriented Gradients)** features
- Clean separation between training and evaluation workflows

## How to install
### 1. Clone the repository
```bash
git clone https://github.com/coldoctopus/FER-with-HOG-and-SVM.git
cd facial-expression-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Activate the enivronment
On Windows:
```bash
.venv\Scripts\activate
```

On Mac/Linux:
```bash
source .venv/bin/activate
```
### Note: Auto-activate virtual environment in VSCode
Once you've created .venv/ and opened VSCode:
- Open the Command Palette (Ctrl+Shift+P)
- Select "Python: Select Interpreter"
- Pick your .venv/ environment
Now VSCode will use it automatically.

### 3. Prepare the dataset
A sample train dataset (sample_1) has been prepared for all users. You can add your own datasets in the /datasets folder.<br>
Please remember to put all data in 5 separate sub-folders, anger, happy, neutral, sad, suprise.

## How to run
### 1. Model training and validation
If you want to train a model yourself, you can use the notebook train_model.ipynb and change the parameters as you please.<br>
After training, your model can be saved inside the /models folder by running the cell at the end of train_model.ipynb.<br>
For validation, you can use the notebook evaluate_model.ipynb. This notebook can be used to run your saved model.  

