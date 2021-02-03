# ExactMEMES 

Implementation of MEMES framework with ExactGP as surrogate function.

## Steps to run framework
Step 1. Store the list of smiles in text file with. (See example for format - `../DeepMEMES/Example/all.txt`.)

Step 2. Store the features of corresponding smile in `pickle format`. (See example for format - `../DeepMEMES/Example/features_temp.pkl` directory. Only a subset is given in Examples)

Step 3. Cluster the dataset into `n` clusters. Store cluster of each in format `../DeepMEMES/Example/labels20.txt` (Script to be added; See example for format in `Example` directory)

Step 4. Edit the scoring function in `memes.py`

Step 5. Run the framework
        `python memes.py --run 1 --rec 4btk --cuda 0 --feature mol2vec` 
        
#### Parameters
   `--iters` number of iterations of BO. Default `40`
   
   `--run`   number of iterations of BO. Required
   
   `--cuda`  GPU number to be used. If you want to use CPU pass `cpu`
   
   `--rec`  receptor name / any name can be given
   
   `--feature` featurization technique used to featuirze molecule
   
   `--features_path` Path to precomputed feature
   
   `--capital` maximum number of evaluations of property allowed. Default `15000`
   
   `--initial` size of initial population. Default `5000`
   
   `--n_cluster` number of clusters made. (Note: Filename of label{n}.txt should match with number of clusters passed as parameter).
   
   `--periter` number of molecules to be sampled in every iteration. Default `500`
   
   `--eps` Exploration parameter. Default `0.05`
   
