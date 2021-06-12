# Fine_tune_BERT_with_spacy3

This is the code to reproduce the results shown in tutorial: https://colab.research.google.com/drive/1g9wV7Teg03BQuJ3t_8QHfuFbyTgUcoTK

First we convert the IOB file exported from the UBIAI annotation tool to spacy JSON:
```
!python -m spacy convert drive/MyDrive/train_set_bert.tsv ./ -t json -n 1 -c iob
!python -m spacy convert drive/MyDrive/dev_set_bert.tsv ./ -t json -n 1 -c iob
```
After converting the training and dev files to JSON file, we need to convert them to spacy binary file:
```
!python -m spacy convert drive/MyDrive/train_set_bert.json ./ -t spacy
!python -m spacy convert drive/MyDrive/dev_set_bert.json ./ -t spacy
```
Next we install spacy and transformer library pipeline:

```
pip install -U spacy
!python -m spacy download en_core_web_trf
```
Next we install the cuda:
```
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-9.2
```
Install pytorch:
```
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Install spacy and transformer packages:
```
!export CUDA_PATH="/usr/local/cuda-9.2"
!pip install -U spacy[cuda92,transformers]
```
Install cupy
```
!pip install cupy
```
Set the cuda path
```
!export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```
After updating the spacy config.cfg file with the training and test paths, we auto-fill the config file with the rest of the parameters that the BERT model will need
```
!python -m spacy init fill-config drive/MyDrive/config.cfg drive/MyDrive/config_spacy.cfg
```
Before launching the training, lets debug the config file to make sure everything is correct:
```
!python -m spacy debug data drive/MyDrive/config.cfg
```
Finally, we are ready to start the training:
```
!python -m spacy train -g 0 drive/MyDrive/config.cfg — output ./
```
After training, the model will be saved in a folder named model-best. Lets try to extract entities using the newly trained model:
```
nlp = spacy.load(“./model-best”)
text = [
'''Qualifications
- A thorough understanding of C# and .NET Core
- Knowledge of good database design and usage
- An understanding of NoSQL principles
- Excellent problem solving and critical thinking skills
- Curious about new technologies
- Experience building cloud hosted, scalable web services
- Azure experience is a plus
Requirements
- Bachelor's degree in Computer Science or related field
(Equivalent experience can substitute for earned educational qualifications)
- Minimum 4 years experience with C# and .NET
- Minimum 4 years overall experience in developing commercial software
'''
]
for doc in nlp.pipe(text, disable=["tagger", "parser"]):
    print([(ent.text, ent.label_) for ent in doc.ents])
```
If you have any questions or run into issues just email us at admin@ubiai.tools.
    
