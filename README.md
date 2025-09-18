Download Python version 3.11.0 from:
https://www.python.org/downloads/release/python-3110/

and install it.

Download the dataset from:
https://www.kaggle.com/datasets/hearfool/vggface2
,
extract the files, and place the dataset into the folder:
Final_ProJect_AI\FaceDetection

Rename the dataset folder to dataset.

Open VS Code.

Press Ctrl+Shift+P → type "Python: Select Interpreter"

Select Python 3.11.0

Open a terminal by selecting Terminal → New Terminal or pressing Ctrl+Shift+~

Create a virtual environment:

python -m venv venv


Activate the virtual environment:

.\venv\Scripts\activate


Install all required packages from requirements.txt:

pip install -r requirements.txt


If you encounter an Execution Policy Error while activating venv, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


Then create the virtual environment again if it hasn’t been created yet:

python -m venv venv
.\venv\Scripts\activate


Reinstall the dependencies:

pip install -r requirements.txt


After that, you can run the program normally.
