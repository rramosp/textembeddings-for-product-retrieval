import pandas as pd
import numpy as np


import numpy as np
import pandas as pd
from tqdm import tqdm
from google import genai
from openai import OpenAI
from time import sleep
from joblib import Parallel, delayed

from google.genai.types import EmbedContentConfig


def compute_ndcg(df):
    """
    df: a pandas dataframe with columns 'query_id, product_id, relevance, model_rank' where
        for each query id, there is a set of rows containing the products retrieved by the model
        together with the rank position assigned by the model. Assumes model_rank starts in 1 with 
        consecutive numbers as the rank increases.
        the column 'relevance' is the ground truth, representing the relevance of each product
        to each query. higher numbers are better.

    returns:
        df_dcg: a dataframe with the same number of rows as df, with two additional columns
                'dcg': the dcg of the rank produced by the model to each query
                'idcg': the ideal dcg ranking by relevance for each query
        ndcg: a dataframe with one column per query containing them sum dcg and idcg
              of the ranked items for each query, together with ndcg = dcg / idcg in [0,1]
    """
    
    bri = []
    for q in tqdm(np.unique(df.query_id.values)):
    
        bq = df[df.query_id==q].sort_values(by='model_rank')
    
        # add ideal rank (as of using ground truth)
        bqi = bq.sort_values('relevance', ascending=False)
        bqi['ideal_rank']=np.arange(len(bqi))+1
        bqi = pd.merge(bq, bqi[['product_id', 'ideal_rank']], how='inner', left_on='product_id', right_on='product_id')    
        bri.append(bqi)
    
    bri = pd.concat(bri)
    
    # copmute ndcg following https://medium.com/data-science/demystifying-ndcg-bee3be58cfe0
    bri['dcg'] = bri.relevance / np.log2(bri.model_rank+1)
    bri['idcg'] = bri.relevance / np.log2(bri.ideal_rank+1)
    
    ndcg = bri.groupby('query_id')[['dcg', 'idcg']].sum()
    ndcg['ndcg'] = ndcg.dcg / ndcg.idcg

    df_dcg = bri
    return df_dcg, ndcg


def load_esci_examples(ESCI_DATASET_ROOT, locale='es'):

    """
    ESCI_DATASET_ROOT: path where https://github.com/amazon-science/esci-data was cloned
    """
    
    df_examples = pd.read_parquet(f'{ESCI_DATASET_ROOT}/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
    df_products = pd.read_parquet(f'{ESCI_DATASET_ROOT}/shopping_queries_dataset/shopping_queries_dataset_products.parquet')
    df_sources = pd.read_csv(f"{ESCI_DATASET_ROOT}/shopping_queries_dataset/shopping_queries_dataset_sources.csv")
    
    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale','product_id'],
        right_on=['product_locale', 'product_id']
    )
    
    # ground truth
    esci_label2relevance_pos = {
        "E" : 4,
        "S" : 2,
        "C" : 3,
        "I" : 1,
    }
    
    dgt = df_examples_products[df_examples_products["large_version"] == 1].copy()
    
    # keep only spanish queries
    dgt = dgt[dgt.product_locale==locale].copy()
    
    dgt['relevance'] = dgt['esci_label'].apply(lambda esci_label: esci_label2relevance_pos[esci_label])
    return dgt


def get_esci_product_string(row):
    """
    row: a row from the dgt dataframe
    """
    di = row
    s = f"<TITLE>{di.product_title}</TITLE>"
    if di.product_description is not None:
        s += f"\n<DESCRIPTION>{di.product_description}</DESCRIPTION>"
    if di.product_bullet_point is not None:
        s += f"\n<BULLETS>{di.product_bullet_point}</BULLETS>"
    if di.product_brand is not None:
        s += f"\n<BRAND>{di.product_brand}</BRAND>"
    if di.product_color is not None:
        s += f"\n<COLOR>{di.product_color}</COLOR>"
    
    return s


def get_wands_product_string(row):
    """
    row: a row from the dgt dataframe
    """
    di = row
    s = ''
    if di.product_name is not None:
        s = f"<NAME>{di.product_name}</NAME>"
    if di.product_class is not None:
        s += f"\n<CLASS>{di.product_class}</CLASS>"
    if di['category hierarchy'] is not None:
        s += f"\n<CATEGORY_HIERARCHY>{di['category hierarchy']}</CATEGORY_HIERARCHY>"
    if di.product_description is not None:
        s += f"\n<DESCRIPTION>{di.product_description}</DESCRIPTION>"
    if di.product_features is not None:
        s += f"\n<FEATURES>{di.product_features}</FEATURES>"
    
    return s


def get_model_ranking(ranking_csv, dgt):
    """
    ranking_csv: csv file with two columns: query_id, product_id
                 with several rows for each query, with products ranked according
                 to the model product predictions for that query
    dgt: result from load_examples
    """
    
    b = pd.read_csv(ranking_csv)
    # join with ground truth
    b = pd.merge(b, dgt, how='left', left_on = ['query_id', 'product_id'], right_on = ['query_id', 'product_id'])#[['query_id', 'product_id', 'relevance']]

    # append rank position as predicted by model
    br = []
    for q in tqdm(np.unique(b.query_id.values)):
        bq = b[b.query_id==q].copy()
        bq['model_rank'] = np.arange(len(bq))+1
        br.append(bq)
    
    br = pd.concat(br)
    return br


def get_gemini_embeddings(texts_list, model, project_id, location, task_type, n_jobs=-1, max_tries=20):

    """
    texts_list: a list of n strings 

    return: a np array of n rows, with the embeddings corresponding to each string in texts_list
    """

    if not isinstance(texts_list, list):
        raise ValueError('texts_list must be a list of strings')

    texts_list = [t.replace("\n", " ") for t in texts_list]

    def f(text):
        current_try = 1
        
        while True:
            try:
                client = genai.Client(vertexai=True , project=project_id, location=location)
                result = client.models.embed_content(
                        model=model,
                        contents=[text],    
                        config=EmbedContentConfig(
                            task_type=task_type
                        )
                )
                return np.r_[result.embeddings[0].values]
            except Exception as e:
                if np.random.random()<.1:
                    print ('.', end='', flush=True)
                if current_try > max_tries:
                    raise e

                sleep(5)
                current_try += 1

    r = Parallel(verbose=1, n_jobs=n_jobs)(delayed(f)(text) for text in texts_list)
    
    
    return np.r_[r]


def get_openai_embeddings(texts_list, api_key, model='text-embedding-3-large'):

    """
    texts_list: a list of n strings 

    retruns: a np array of n rows, with the embeddings corresponding to each string in texts_list
    """

    if not isinstance(texts_list, list):
        raise ValueError('texts_list must be a list of strings')

    texts_list = [t.replace("\n", " ") for t in texts_list]

    client = OpenAI(api_key = api_key)
    
    result = client.embeddings.create(input = texts_list, model=model)

    if not 'data' in dir(result):
        raise ValueError(f'error in request {result}')
    
    r = np.r_[[ri.embedding for ri in result.data]]
    return r


def get_batch_embeddings(texts, embeddings_fn, api_key, model, chunk_size=100):
    texts = list(texts)
    
    embeddings = []
    n_chunks = len(texts) // chunk_size

    for i in tqdm(range(0, len(texts),chunk_size)):
        embs = embeddings_fn(texts[i:i+chunk_size], api_key, model)
        for ei in embs:
            embeddings.append(ei)

        # sleep to keep down to quota maximun rate per minute
        sleep(.1)
    
    return embeddings


def get_openai_batch_embeddings(texts, OPENAI_API_KEY, model='text-embedding-3-large',  chunk_size=100):
    return get_batch_embeddings(texts, get_openai_embeddings, OPENAI_API_KEY, model, chunk_size=chunk_size)

