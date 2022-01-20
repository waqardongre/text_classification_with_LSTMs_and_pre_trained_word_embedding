## This is text classification project with LSTMs and pre-trained word embeddings like glove.6B.zip.
## Developed a RESTful server/API which can perform predictions on the given input text in the respective API html form using the model. 
## The development and deployment of the program is a Linux environment. Used WSL Ubuntu for Windows to develop and run this program.





## Below are the instructions to run it in your localhost.

### Download and extract if needed my project repository.

### Considering you have a linux based terminal
### Run below in your linux based terminal to install uvicorn if not installed else skip
pip install fastapi uvicorn

### Run below in your linux based terminal  to install tensorflow if not installed else skip
pip install tensorflow==2.7.0

### Go to the project directory in in your linux based terminal
cd /mnt/c/Users/<username>/path_to_your_project

### To run the developed rest api, run below command in in your linux based terminal 
uvicorn app:app --reload

### It will show the hosted link in output, copy and paste it in your browser like 'http://127.0.0.1:8000' you have to add '/predict' after it so it become 'http://127.0.0.1:8000/predict' and you will see the form in your browser which will predict the text which you will enter in the given textbox. to predict results you have to just click on the 'predict' submit button. Please check screenshots share with the project repostory.