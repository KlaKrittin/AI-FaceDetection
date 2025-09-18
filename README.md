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

If you see an Execution Policy Error while activating venv, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then recreate and activate venv: python -m venv venv 
                                 .\venv\Scripts\activate

Reinstall dependencies:

  pip install -r requirements.txt

7. Run the Program

  Now you can run the program normally 🎉


