
# text embeddings for product retrieval

Use text embeddings to represent both queries and products, and compute the dot product to get the most similar products given a query.

## Datasets


We use three datasets

- the [esci](https://github.com/amazon-science/esci-data) Spanish dataset

- the [esci](https://github.com/amazon-science/esci-data) English dataset

- the [wands](https://github.com/wayfair/WANDS) dataset which is in English

The following tables summarizes the datasets

|   | wands  | esci-es  | esci-us  |
|---|---|---|---|
|num_products | 42994 | 259973 | 1215851 |
|num_queries | 480 | 15180 | 97345 |
|num_relevance_judgements | 233448 | 356410 | 1818825 |
| mean_num_relevance_judgements_per_query | 486.3 | 23.5 | 18.7 



## Results

 We use binary relevance, considering in `wands` other than relevance 2 as not relevant, and in `esci-us` and `esci-es` other than relevance 4 as not relevant (see the distribution of relevance judgements below).

With [`sklearn ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html), using `1/ranked_position` as given by dot product similarity as `y_score`

|	| wands|	esci-es	|esci-us|
|---|---|---|---|
|google `text-embedding-large-exp-03-07`|	0.638	|0.697	|0.651|
|openai	`text-embedding-3-large`|0.615|	0.676|	0.623|
|google `text-embedding-004`	|0.618	|0.554	|0.606|



## Notes

- `text-004` are only English embeddings however they perform reasonably well on `esci-es`. See [notebook 05](https://github.com/rramosp/textembeddings-for-product-retrieval/blob/main/05%20-%20inspect%20ranking.ipynb).

- relevance judgement distribution on `wands`

|relevance|label| pct of judgements|
|---|---|---|
|2 |Exact |    0.109720|
|1 |Partial|    0.628118|
|0 |Irrelevant|    0.262161|

- relevance distributions on `esci`


|relevance|label|esci-us|esci-es|
|---|---|---|---|
|4 | Exact |   0.566948 | 0.685914
|3 | Substitute |  0.057526 | 0.022019
|2 | Complemente |  0.249878  | 0.203050
|1 | Irrelevant |   0.125647 | 0.089016

