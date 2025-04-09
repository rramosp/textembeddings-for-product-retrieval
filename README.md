
# text embeddings for product retrieval

Use text embeddings to represent both queries and products, and perform dot product to get the most similar products given a query.

We use the [esci](https://github.com/amazon-science/esci-data) dataset, with the following locales

    locale es (spanish) 15180 queries, 259973 products
    locale us (english) 97345 queries, 1215851 products

The dataset contains `(query, [products])` pairs so that for each query there is a list of products each one manually labeled as Exact, Substitute, Complement or Irrelevant with respective relevance scores 4,3,2,1 when computing nDCG.


Results with Gemini `text-embedding-004`, embedding size 768

                    full    only annotated
    spanish nDCG   0.6664       0.9680
    english nDCG   0.6765       0.9785

Results with OpenAI `text-embedding-3-large`, embedding size 3072

                    full    only annotated
    spanish nDCG   0.7384       0.9724
    english nDCG   0.6933       0.9785

Dataset baseline (performed on a test split, includes Japanese)

                            only annotated
    global nDCG                 0.83


**full** experiments search in all existing products given a query. This effectively results in many false negatives (see comment below).

**only annotated** experiments search only in the products that have been annotated for each query. For a given query, this effectively amounts to having the retrieval model (embeddings dot product) trying to rank the products annotated for that query without looking at any other products.

## retrieval inspection on full experiments

The dataset has nice annotations but there seems to be many not-annotated products that may also be relevant to queries. This affects the **full** experiments which do retrieval in the full product space. In the following three query examples there are many products that clearly match the query but have not been identified as relevant. Comparisons between models in this sense have to be interpreted with care.![alt text](imgs/queries.png) 

## product embeddings

Products are described by the following fields `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand` and `product_color`. All products have id and title, but many are missing one or more of the other fields. Existing fields are concatenated with corresponding XML tags and then sent to generate embeddings. 

The following is an example of the string assembled for one random product

![alt text](imgs/product.png)