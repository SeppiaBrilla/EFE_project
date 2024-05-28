# Feature generator

This script allows to generate new instances starting from an instance file. The subfolder ```feature generators``` contains the classes that create the actual features.
Here are the possible choices:
- ```--type``` (dnn/fzn2feat): The type of features to get.
- ```--instance```: The instance file to use. In json format for the dnn features and in essence format for fzn2feat
- ```--names```: The name to use for the dnn probability output. Ignored for fzn2feat
- ```--probability-only```: If used, the dnn features will contain only the probability values of the neural network output
- ```--weights```: required for the dnn option: the weights used by the neural network
- ```--eprime```: required for the fzn2feat option: the eprime file to use to predict the features
- ```--output``` (json/csv): the output format of the script
- ```--time```: if true, the script outputs the time required to produce the features  
