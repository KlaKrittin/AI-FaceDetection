1. Install Python

          Download and install Python 3.11.0 from:
          👉 Python 3.11.0 Download

2. Download Dataset

            Download the dataset from VGGFace2 Dataset
          
            Extract the files
          
            Move the dataset into: Final_ProJect_AI/FaceDetection
          
            Rename the folder to: dataset

3. Setup VS Code

            Open VS Code
          
            Press Ctrl + Shift + P → type Python: Select Interpreter
          
            Select Python 3.11.0

4. Create Virtual Environment

            Open terminal (Ctrl + Shift + ~) and run: python -m venv venv
          
            Activate the environment: .\venv\Scripts\activate

5. Install Dependencies
            pip install -r requirements.txt

6. Fix Execution Policy Error (if needed)

          If you see an Execution Policy Error while activating venv, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
          
          Then recreate and activate venv: python -m venv venv 
                                           .\venv\Scripts\activate

          Reinstall dependencies: pip install -r requirements.txt

  Now you can run the program normally 🎉

  Training Results
  
Performance (F1 / Precision / Accuracy) by Augmentation Type
             
<img width="1200" height="600" alt="augmentation_bar_chart" src="https://github.com/user-attachments/assets/f672828b-406b-428b-8273-0c566ac1e9da" />

Confusion Matrix
<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/8783cdee-5843-44a3-afed-258f0e55d291" />

ROC Curve
<img width="640" height="480" alt="roc_curve" src="https://github.com/user-attachments/assets/bdc28d68-a574-486b-a162-d8e0cd741f1d" />


