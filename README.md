This repository is for the MS MARCO http://www.msmarco.org/<br>
Fixed some bugs and edited original files from https://github.com/dfcf93/MSMARCOV2/tree/master/



# BiDaF

### Requirements
~~~
Python 3.5
CUDA 9.0      # when usng GPU
Pytorch       # prefer version > 0.3 
h5py
nltk
spacy
~~~
### Conda virtual environment setup (Optional)

#### Create new environment
~~~
conda create -n <env_name>                                         # change <env_name>, name of your environment
conda create -n <env_name> python=3.5 h5py nltk jupyter notebook   # you can add packages
~~~
#### Activate environment before usage
~~~
source activate <env_name>            # for linux, osx
activate <env_name>                   # for window
~~~
#### Install pytorch
~~~
conda install pytorch                 # for installing latest version (linux-64, osx-64, win-64)
conda install -c soumith pytorch      # pytorch 0.3.1 (linux-64, osx-64)
conda install -c peterjc123 pytorch   # pytorch 0.3.1 (win-64)
~~~
#### Install Spacy
~~~
pip install spacy
~~~
then install the model from spacy
~~~
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.0.0/en_core_web_lg-2.0.0.tar.gz
~~~
#### Useful conda commandline
~~~
conda info --envs                                   # list of virtual env list you have created
conda env listconda list                            # list of packages in the env
~~~
## Setup for BiDaF

#### 1. Ensure the mrcqa folder is in your python Path
~~~
export PYTHONPATH=${PYTHONPATH}:~/<Where you saved this folder>/mrcqa
~~~
For example,
~~~
export PYTHONPATH=${PYTHONPATH}:~/MS_MARCO_Edited/mrcqa
~~~
#### 2. experiment folder has a copy of the config.yaml file from scripts.
Currently, this will only train for 1 epoch and stop. This iis useful for testing and debugging scripts.

#### 3. MSMARCO v2.1 data is in Data folder
Pre-trained word embedings (if needed, e.g. GloVe 42B) save in your Data folder.

#### 4. Try to train a model using the following script
Before running following commandline you have to be in the MS_MARCO_Edited/ folder
~~~
python ./scripts/train.py ./experiment ./Data/<chosen training data> --cuda=False                 # cpu
python ./scripts/train.py ./experiment ./Data/<chosen training data> --cuda=True                  # gpu
python ./scripts/train.py ./experiment ./Data/<chosen training data> --force_restart --cuda=False
~~~
force_restart is not strictly required but it is used to ignore any existing $checkpoints$ in your experiment folder.

If using pre-trained word embeddings,
~~~
python scripts/train.py ./experiment ./Data/<chosen training data> --word_rep ./Data/<chosen word embedding> --force_restart --cuda=False
~~~
#### 5. If your model successfully finnished training then you are ready to start training a full model and experimenting

## Training

#### Modify config.yaml file<br>
##### in experiment folder AND scripts file AND MS_MARCO_Edited file to match your desired paramaters such as training epochs, dropout rate, learning rate etc.

and run the following scripts. (same as 4. above)
~~~
python ./scripts/train.py ./experiment ./Data/<chosen training data> --cuda=False
~~~
### Training Senario 1 -- have checkpoint
##### Check your experiment file and if there is a file called checkpoint without extension (not checkpoint.opt).

#### a) And you want to resume training,
~~~
python ./scripts/train.py ./experiment ./Data/<chosen training data> --cuda=False
~~~
#### b) You dont want to resume training,
~~~
python ./scripts/train.py ./experiment ./Data/<chosen training data> --force_restart --cuda=False
~~~
### Training Senario 2 -- do not have checkpoint
##### Check your experiment file and if there is NO file called checkpoint without extension (not checkpoint.opt).
~~~
python ./scripts/train.py ./experiment ./Data/<chosen training data> --cuda=False
~~~
## Prediction

#### To generate a prediction file run the command below.

This will load the model and data and predict for all answered questions where the answer is a span.<br>
Any new tokens/char will get a random embedding.
~~~
python scripts/predict.py ./experiment ./Data/<chosen data for prediction> prediction.json --cuda=False  # cpu
python scripts/predict.py ./experiment ./Data/<chosen data for prediction> prediction.json --cuda=True   # gpu
~~~
To generate new embeddings from your embedding file instead instead of using random embeddings use the following command. (optional)
~~~
python scripts/predict.py ./experiment ./Data/<chosen data for prediction> prediction.json --word_rep ./Data/<chosen word embedding> --cuda=False
~~~
## Evaluation

Before evaluating the prediction, check MS_MARCO_Edited/data_processing.ipynb#Prediction-data.

#### Then you should be in the Evaluation folder and run following script
~~~
./run.sh ../Data/reference.json ../Data/candidate.json./run.sh ../Data/<generated reference file> ../Data/<generated candidate file>
~~~
### Sometimes, there will be an error like,
~~~
Exception: "{"query_id": 91867, "answers": ["C:\WINDOWS\system32\config\SM Registry Backup."]}
" is not a valid json
~~~
#### Then check the error code, and if it is from
~~~
load_file(p_path_to_reference_file)                                  # ----> Go to ./Data/reference.json
candidate_no_answer_query_ids = load_file(p_path_to_candidate_file)  # ----> Go to ./Data/candidate.json
~~~
##### Find line caused the error by finding the query_id like e.g. "query\_id": 91867 and fix it to the right format
##### Most of the time if you put one more \ (slash) or remove the character causing error or if the data is separated into two lines, then you can fix the error by combining the data in one line.

Do not delete the whole line, then it will cause another error.
Do not use jupyter to open the file, use other editor like sublime text or atom which can detect the wrong json format.

### Output for evaluation will be in the similar format to bellow:
~~~
############################
F1: 1.0
Semantic_Similarity: 0.7561276694449763
bleu_1: 0.4613473375119735
bleu_2: 0.4358699088949773
bleu_3: 0.4222062580786961
bleu_4: 0.41305888231064947
rouge_l: 0.46252234746236
############################
~~~
# Data Structure

## Training Data

#### Training data should look like this, not exactly like this but json pretty, json file which has all of its data in one line doesn't work.
~~~
{
    "query_id": {
        "10": 1102421,     # .values() of query_id dict is not useful
        "100004": 90836,   # only query_id .keys() is the query_id
        ...
        ...
    },
    
    "query": {
        "10": "why did the progressive movement fail to advance racial equality quizlet",
        "100004": "chart for foods low in potassium.",
        ...
        ...
    },
    
    "answers": {
        "10": ["No Answer Present."],
        "100004": ["The average cost of a set of brake pads is above $25."],
        ...
        ...
    },
    
    "passages": {
        "10": [{"is_selected": 0,
                "passage_text": "The Progressive Era spanned the years from 1890 to 1920.",
                "url": "https://www.thoughtco.com/african-americans-in-the-progressive-era-45390"},
               {"is_selected": 0,
                "passage_text": "1 W.E.B Du Bois was the founder of the Niagara Movement and later the NAACP.",
                "url": "https://www.thoughtco.com/african-americans-in-the-progressive-era-45390"},
             ...
             ...
             
               {"is_selected": 0,
                "passage_text": "Du Bois disagreed with Washington.",
                "url": "https://www.thoughtco.com/african-americans-in-the-progressive-era-45390"}],
        "100004": [{"is_selected": 0,
                "passage_text": "Low Sodium Low Potassium Foods List.",
                "url": "http://www.etoolsage.com/chart"},
               {"is_selected": 0,
                "passage_text": "Alleviation of High Blood Pressure (Hypertension) - Studies show that a diet 
                                    high in potassium, especially potassium from fruits and vegetables, lowers 
                                    blood pressure.",
                "url": "https://www.healthaliciousness.com/articles/food-sources-of-potassium.php"},
               {"is_selected": 0,
                "passage_text": "High potassium foods include beans, dark leafy greens, potatoes, squash, yogurt, 
                                    fish, avocados, mushrooms, and bananas. The current daily value for potassium 
                                    is 3,500 milligrams (mg).",
                "url": "https://www.healthaliciousness.com/articles/food-sources-of-potassium.php"},
              ...
              ...
         }
    }
}
~~~
## Prediction

#### After prediction, prediction.json file should look like this
~~~
'10190' 'The average cost of a set of brake pads is $ 25–75 .' 0 50
'56809' 'This math worksheet will help give your Preschool , Kindergarten , or 1st grader some extra practice writing their numbers from 1-20 .' 0 131

...
...
~~~
## Evaluation

However, to evaluate our prediction, prediction.json file has to be reformated and Data/candidate.json and Data/reference.json has be generated. Please refer to the MS_MARCO_Edited/data_processing.ipynb#Prediction-data.

#### After data processing, Data/candidate.json file structure
~~~
{"query_id": 10190, "answers": ["The average cost of a set of brake pads is $ 25–75 ."]}
{"query_id": 56809, "answers": ["This math worksheet will help give your Preschool , Kindergarten , or 1st grader some extra practice writing their numbers from 1-20 ."]}

...
...
~~~
#### Data/reference.json file structure should be like this, same as the Data/candidate.json file
~~~
{"query_id": 10190, "answers": ["The average cost of a set of brake pads is $25–75. Where rotors are $75–150 a piece."]}
{"query_id": 56809, "answers": ["Kindergarten, or 1st grader some extra practice writing their numbers from 1-20."]}

...
...
~~~
