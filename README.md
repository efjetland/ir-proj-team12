# Team-12 SMART-task project github

The project consists of three parts
1. Project.ipynb jupyter notebook to index the document collection and train queries.
2. ranking.py python file for scoring using baseline models, training the ltr and final ranking results.
3. evaluation.py python file provided by the smart task github to evaluate the results from step 2.

## Project.ipynb
Requires a running elasticsearch on localhost to run and index.

## Ranking.py
Requires the indexing to be complete and train and test query files to exist in the data folder(they are created when indexing is run)

## Evaluation.py 
Takes three arguments, a type hierarchy tsv file, a ground truth json file, and the system ranking json file. The type tsv file is provided as dbpedia_types.tsv, also from the SMART task github.
The ground truth and system ranking files are created on ranking, and are named gold.json and system_ranking.json respectively. Ranking also produces three system ranking
files for each of the baseline methods as well. We have included our final ranking results in the repository.
