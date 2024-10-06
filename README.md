# BiologicallyInspiredModelsOfLearning

`[Last update: October  6, 2024]`

***
    Period:     2023-09 - ...
    Status:     work in progress

    Author(s):  Kittelmann
    Contact:    denisekittelmann@gmail.com

***

*In general, one can add README's in nearly every folder. The guiding principle should always be that any person who is not familiar with the project can find their way exclusively via the README's – 'This may be you one day'*

## Project description

This projects investigates the alignment between the brain and Biologically Inspired Artificial Neural Networks in comparison to the alignment between the brain and traditional Artificial Neural Networks (ANNs) by comparing the temporal dynamics evoked by an Predictive Coding Network (PCN) and an backpropagation-trained ANN using Representational Similarity Analysis (RSA).

## Project structure

*A brief description of the folder structure of the project (Where is what?). Anticipate new lab members who suppose to be able to orientate within this structure without your help. At the same time, avoid too detailed descriptions. Down the folder structure, there suppose to be further README's explaining subsequent folders & data.*

## Install research code as package

In case, there is no project-related virtual / conda environment yet, create one for the project:

```shell
conda create -n BiMo_3.9 python=3.9
```

And activate it:

```shell
conda activate BiMo_3.9
```

Then install the code of the research project as python package:

```shell
# assuming your current working directory is the project root
pip install -e ".[develop]"
```

**Note**: The `-e` flag installs the package in editable mode,
i.e., changes to the code will be directly reflected in the installed package.
Moreover, the code keeps its access to the research data in the underlying folder structure.
Thus, the `-e` flag is recommended to use.


*Similarly, use this structure for Matlab or other programming languages, which are employed in this project.*

## Publications

*List publications resulted from this project (including papers, posters, talks, ...)*



## Contributors/Collaborators

*Name people who are involved in this project, their position and/or contribution. Optional: add contact data*
Dirk Gütlin & Ryszard Auksztulewicz
