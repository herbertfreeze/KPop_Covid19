import pandas as pd
import zipfile
import glob
from ast import literal_eval
import numpy as np
from itertools import product
import tldextract

kpop_hashtags = (" #kpop #kpopidol #kpopidols #idol #idols #kpopfanart #kpopfanmade #kpopfans " +
                "#kpopfan #kpopstans #bts #blackpink #blink #twice #KCON #KCON:TACT " +
                "#virtualconcert #fancam #kpopfancam #SMEntertainment #YGEntertainment #JYPEntertainment #JYP")
kpop_hashtags2 = " #once #YG #army #twice"
kpop_hashtags = kpop_hashtags.lower().split(" #")[1:]
kpop_hashtags2 = kpop_hashtags2.lower().split(" #")[1:]

cols = ['tweetid', 'userid', 'screen_name', 'date', 'lang', 'location', 'text',
       'extended', 'reply_userid', 'reply_screen', 'reply_statusid',
       'tweet_type', 'verified', 'hashtag', 'friends_count', 'followers_count','urls_list',
        'rt_hashtag', 'qtd_hashtag', 'sent_vader', 'token', 'media_urls','state', 'country', 'rt_state', 'rt_country',
       'qtd_state', 'qtd_country'
       ]

def get_wearmask(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[df.apply(lambda r: (type(r.hashtag) == str) , axis = 1)]
        df.hashtag = df.hashtag.str.lower()
        # df["text_processed"] = df.text.apply(lambda x: x.lower())
        df = df[df.hashtag.apply(lambda x: "wearamask" in x)]
        df.to_pickle("Processed_Data/0_MASK_DF/{}.pkl".format(index), 
                     compression='gzip')
    except:
        print(index)
        

        
def get_no_mask(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[df.apply(lambda r: (type(r.hashtag) == str) , axis = 1)]
        df.hashtag = df.hashtag.str.lower()
        # df["text_processed"] = df.text.apply(lambda x: x.lower())
        df = df[df.hashtag.apply(lambda x: "masksoff" in x)]
        df.to_pickle("Processed_Data/0_NO_MASK_DF/{}.pkl".format(index), 
                     compression='gzip')
    except:
        print(index)
        
        
def get_tedros(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[df.apply(lambda r: (type(r.text) == str) , axis = 1)]
#         df["text_processed"] = df.text.apply(lambda x: x.lower())
        df = df[df.text.apply(lambda x: "RT @DrTedros" in x)]
        df.to_pickle("Processed_Data/0_Tedros/{}.pkl".format(index), 
                     compression='gzip')
    except:
        print(index)
        
H = pd.read_csv("cleanedv0819.csv")
H = H[H.Type =="Health"]
H = H.user_id.astype(np.int64).values

def get_health_agencies(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[df.apply(lambda r: (type(r.text) == str) , axis = 1)]
        df = df[(df.userid.isin(H)) | (df.rt_userid.isin(H))]
        df.to_pickle("Processed_Data/0_Health/{}.pkl".format(index), 
                     compression='gzip')
    except:
        print(index)
        
def read_multicore(f):
    try:
        df = pd.read_pickle(f)
    except:
        try:
            df = pd.read_pickle(f, compression = "gzip")
        except:
            print(f)
            return f
    return df

a = read_multicore("Processed_Data/0_State_Count/0.pkl")
a.sort_values("state", inplace = True)
state_cols = a.state.values

def get_state_count(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[["state"]]
        df = df[~df.state.isna()]
        ST = df.state.value_counts().reset_index(drop=False)
        ST.columns = ["state", "num"]
        ST.to_pickle("Processed_Data/0_State_Count/{}.pkl".format(index), 
                     compression='gzip')
    except:
        print(index)

def multicore_read_and_sort(f):
    df = read_multicore(f)
    df.sort_values("state", inplace = True)
    df.set_index('state', inplace=True)
    df = df.reindex(index=state_cols,fill_value = 0, )
    return df.values 

#     return df[cols]

def unzip_load_worker(p):
    i,zip_file, file = p
    zf = zipfile.ZipFile(zip_file)
    return pd.read_csv(zf.open(file),
                 lineterminator='\n')

def is_kpop(x): # tags as words
    for k in kpop_hashtags:
        if k in x:
            return k
    return ""

def is_kpop_ht(x): # tags as hashtags
    for k in kpop_hashtags2:
        if k == x:
            return True
    return False

def try_eval(s):
    try:
        return literal_eval(s)
    except:
        return np.nan

def get_kpop(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
    #     ## Hashtag basis
    #     df = df[(df["hashtag"] != "[]") & (~df.hashtag.isna())]
    #     df["hashtag_processed"] = df.hashtag.apply(lambda x: literal_eval(x.lower()))
    #     mask = df.hashtag_processed.apply(is_kpop_ht)
        #############
        # Text_Basis
        df = df[df.apply(lambda r: (type(r.text) == str) & (type(r.hashtag) == str) , axis = 1)]
        df["text_processed"] = df.text.apply(lambda x: x.lower())
        df["kw"] = df.text_processed.apply(is_kpop)
        mask = df.kw.apply(lambda x: x != "")
        df["hashtag_processed"] = df.hashtag.apply(lambda x: try_eval(x.lower()))
        df = df[~df.hashtag_processed.isna()]
        mask += df.hashtag_processed.apply(is_kpop_ht)
        mask = mask.astype(bool)
        ##############

        df = df[mask]
        df.to_pickle("Processed_Data/0_Primary_Dataframe/{}.pkl".format(index), compression='gzip')
        return df
    except:
        print(index)
        return None
    
def get_vax(p):
    index, __, __ = p
    try:
        df = unzip_load_worker(p)
        df = df[df.apply(lambda r: (type(r.text) == str) & (type(r.hashtag) == str) , axis = 1)]
        df["text_processed"] = df.text.apply(lambda x: x.lower())
        df["kw"] = df.text_processed.apply(is_kpop)
        mask = df.kw.apply(lambda x: x != "")
        df["hashtag_processed"] = df.hashtag.apply(lambda x: try_eval(x.lower()))
        df = df[~df.hashtag_processed.isna()]
        mask += df.hashtag_processed.apply(is_kpop_ht)
        mask = mask.astype(bool)
        ##############

        df = df[mask]
        df.to_pickle("Processed_Data/0_Primary_Dataframe/{}.pkl".format(index))
        return df
    except:
        print(index)
        return None
    
    
def chain_lists_string(s):
    b = s.apply(process_list_as_string)
    b = b[b != ""]
    b = ", ".join(b)
    return "[" + b + "]"

def process_list_as_string(x):
    return x[1:-1]

def process_urls(url_list):
    samp = url_list
    samp = literal_eval(samp)
    samp = [u["display_url"] for u in samp]
    samp = [tldextract.extract(x).domain for x in samp]
    return samp

def process_url_worker(df):
    return df.urls_list.apply(process_urls)

def parallelize_dataframe(df, func, n_cores=24):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_urls(p):
    try:
        i, zf, f = p
        platform_users = np.load("Processed_Data/bts_users.npy", allow_pickle=True)

        zf = zipfile.ZipFile(zf)
        df = pd.read_csv(zf.open(f), usecols = ["userid", "urls_list"], error_bad_lines=False,warn_bad_lines=True,  encoding='Latin-1',
                 lineterminator='\n')
        df = df[df.urls_list != "[]"]
        df = df[~df.urls_list.isna()]
        df = df[df.userid.apply(lambda x: type(x) == int)]
        df.userid = df.userid.astype(np.int64)
        
        df = df[df.userid.isin(platform_users)]
        B = df.groupby("userid")
        B = pd.DataFrame( B.urls_list.agg(chain_lists_string) )
        B.reset_index(inplace = True)
        B.urls_list = B.urls_list.apply(process_urls)
        B.to_pickle("Processed_Data/Temp/urls1/{}.pkl".format(i))
    except Exception as e:
        print(e)
        print(p)
        return None

    
    