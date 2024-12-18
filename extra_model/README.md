# [F24] Data and Knowledge Representation - Extra Graph Approach (GA) Assignment

## Repository structure

The folder includes the following key files:

- **feature_gen.py**: A script to generate the features for the dataset.
- **evaluation.py**: A script for evaluating the crafted features.
- **experiments.ipynb**: Jupyter notebook with various experiments and statistical visualizations.

## Features engineered

For the purposes of link-prediction in knowledge graph, the following five graph-based features were crafted:

1. **Preferential Attachment (PA)**: A measure of the probability that two nodes are connected, based on their degree.
2. **Clustering Coefficient (CC)**: A measure of how connected the neighbors of a node are.
3. **Adamic-Adar Index (AA)**: A similarity measure that assigns more weight to rare common neighbors between two nodes.
4. **Ruzicka Similarity (RZ)**: A similarity measure based on shared neighbors, weighted modification of Jaccard Index (IoU).
5. **Weighted Overlap Score (WOS)**: A measure of the similarity between two nodes based on their overlapping neighbors, with weights for each common neighbor.

These features were computed over the graph data provided in the original repository.

## Difficulties and solutions

- **1: Crafting meaningful features within a short timeframe and low computational power**  
  Solution: traditional and long established graph-based features that are known to capture various aspects of nodes and links.

- **2: Extremely sparse graph**  
  Solution: **scipy.sparse** and its instruments make wonders in terms of matrix and row/column operations.

- **3: Evaluation**  
  Solution: **ROC-AUC** was used as the evaluation metric to assess the performance of the features when applied both independently and via classic (**Logistic Regression** and **Naive Bayes**) models.

## Evaluation results

### 1. Evaluating features separately via Ranking task
The following table shows the ROC-AUC scores for individual features when they are used to rank potential link candidates:

| Feature                   | ROC AUC    |
|---------------------------|------------|
| Preferential Attachment (PA)   | 0.9422     |
| Clustering Coefficient (CC)    | 0.9380     |
| Adamic-Adar index (AA)        | 0.9495     |
| Ruzicka similarity (RZ)       | 0.7638     |
| Weighted Overlap Score (WOS)  | 0.9288     |

### 2. Evaluating features using classic models as aggregators

| Model                       | ROC AUC    |
|-----------------------------|------------|
| Logistic Regression (PA + CC + AA + RZ + WOS) | 0.9436     |
| Naive Bayes (PA + CC + AA + RZ + WOS)  | 0.9459     |

## Running the Code

To generate features and run evaluation using the graph data, follow these steps:

### 1. Environment setup
Python version: 3.10

Install the required libraries using the following (from the project root):

```bash
pip install -r requirements.txt
```
### 2. Place the dataset in the project root
Download the competition dataset `SemanticGraph_delta_5_cutoff_25_minedge_3.pkl` and place it in the root directory of the repository.

### 3. Run `feature_gen.py` script to generate the features.

### 4. Run `evaluation.py`.

### 5. Observe the results!
Note that combined feature performance (see above) did not go higher than individual, indicating there are possibly other methods needed, e.g. GNNs with downstream feature support instead of classic ML models.