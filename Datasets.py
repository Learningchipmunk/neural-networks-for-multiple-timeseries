import os
import torch
import requests
import json
import datetime
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import Dataset
from numerapi import NumerAPI
from datasets import load_dataset
import requests
import datetime

with open('api_keys.json') as f:
    api_keys = json.load(f)
EODHD_news_api_key = api_keys['EODHD_news_api_key']
NewsAPI_key = api_keys['NewsAPI_key']	

class NumeraiData:
    '''Downloads the latest Numerai data using the NumerAPI + Loads them using pandas. Example usage:

        train_dataset_pt = NumeraiData(data_dir= "NumeraiData/v4.2/", version="v4.2")

    Inspired by https://colab.research.google.com/drive/1W90xCrRE0_MWVvK-wugDM_HhkjgGNXDa?usp=sharing&source=post_page-----792e5960e287--------------------------------#scrollTo=rfyyhPODrPsK
    and https://github.com/numerai/example-scripts/blob/master/hello_numerai.ipynb
    '''
    def __init__(self, data_dir, version="v4.2"):
        self.data_dir = data_dir
        self.napi = NumerAPI(verbosity="info")
        self.version = version

        self.feature_cols = None
        self.target_names = None
        self.train = None
        self.validation = None
        self.live = None
        self.meta_model = None
        self.validation_example_preds = None

    def __download__(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if self.napi.check_new_round():
            print("A new round had started. It is a good idea to check which version of the data is used right now @ https://numer.ai/data/.")
        self.napi.download_dataset(self.version + "/features.json", self.data_dir + "/features.json")
        self.napi.download_dataset(self.version + "/train_int8.parquet", self.data_dir + "/train_int8.parquet")
        self.napi.download_dataset(self.version + "/validation_int8.parquet", self.data_dir + "/validation_int8.parquet")
        self.napi.download_dataset(self.version + "/live_int8.parquet", self.data_dir + "/live_int8.parquet")
        ## Predictions provided by numerai
        self.napi.download_dataset(self.version + "/meta_model.parquet", self.data_dir + "/meta_model.parquet")
        self.napi.download_dataset(self.version + "/validation_example_preds.parquet", self.data_dir + "/validation_example_preds.parquet")

    def load_data(self):
        self.__download__()

        # Load only the "medium" feature set to reduce memory usage and speedup model training (required for Colab free tier)
        # Use the "all" feature set to use all features
        feature_metadata = json.load(open(self.data_dir + "/features.json"))
        self.feature_cols = feature_metadata["feature_sets"]["medium"]

        # Store live data
        self.live = pd.read_parquet(self.data_dir + "/live_int8.parquet")

        # Store meta model predictions
        self.meta_model = pd.read_parquet(self.data_dir + "meta_model.parquet")
        self.validation_example_preds = pd.read_parquet(self.data_dir + "validation_example_preds.parquet")


        cols_to_ignore = [
            c for c in self.live.columns if (c not in self.feature_cols and "feature_" in c)
        ]
        cols_to_load = [c for c in self.live.columns if c not in cols_to_ignore]

        # Store target names
        self.target_names = [c for c in cols_to_load if 'target' in c]

        # Reading Train and Validation data
        self.train = pd.read_parquet(self.data_dir + "/train_int8.parquet", columns=cols_to_load)
        self.validation = pd.read_parquet(self.data_dir + "/validation_int8.parquet", columns=cols_to_load)

        # Converting Eras to ints
        self.era_to_int = {era: i+1 for i, era in enumerate(np.append(self.train['era'].unique(), self.validation['era'].unique()))}
        self.train["era_int"] = self.train["era"].apply(lambda x: self.era_to_int[x])
        self.validation["era_int"] = self.validation["era"].apply(lambda x: self.era_to_int[x])

        ## Preprocessing!
        # Filling NaNs with 0.5, which is the mean for each target
        self.train[self.target_names]      = self.train[self.target_names].fillna(0.5)
        self.validation[self.target_names] = self.validation[self.target_names].fillna(0.5)

        # Splitting test and validation data
        self.test        = self.validation[self.validation.data_type == 'test']
        self.validation  = self.validation[self.validation.data_type == 'validation']

        return self.train, self.validation, self.test, self.live, self.meta_model, self.validation_example_preds
    
    @staticmethod
    def downsample_data_by_era(data, k:int=4):
        '''Takes a dataframe (data) with eras as input. Returns a downsampled dataframe with every kth era (k:int, defautls to 4.).
        '''
        # Downsample to every kth era to reduce memory usage and speedup model training
        dwnspl = data[data["era"].isin(data["era"].unique()[::k])].copy()

        # Converting Eras to ints, taking into account the downsampled eras
        era_to_int = {era: i+1 for i, era in enumerate(dwnspl['era'].unique())}
        dwnspl["era_int"] = dwnspl["era"].apply(lambda x: era_to_int[x])

        return dwnspl

# This dataset loads one era at a time as a sequence.
# You should modify this if you want to load multiple eras (like multiple sequences in sentence pair classification) at once.
class NumeraiDataset(Dataset):
    '''Defines the Dataset class for numerai data. Example usage:

        train_dataset_pt = NumeraiData(train, feature_cols, target_names, max_seq_len, PADDING_VALUE, add_padding=True)

    '''
    def __init__(self, dataframe, feature_names, target_names, max_seq_len, padding_value, era_col_name='era_int', multiple_eras_in_sequence=False, add_padding=True):
        self.data = dataframe
        self.feature_names = feature_names
        self.target_names  = target_names
        self.max_seq_len   = max_seq_len
        self.padding_value = padding_value
        self.era_col_name  = era_col_name
        self.multiple_eras_in_sequence = multiple_eras_in_sequence
        self.add_padding   = add_padding
        self.min_era       = self.data[era_col_name].values.min()

    def __len__(self):
        if self.multiple_eras_in_sequence:
            return  int(len(self.data) / self.max_seq_len)+1
        else:    
            return len(self.data.groupby(self.era_col_name))

    def __getitem__(self, index):
        ## For multiple eras:
        if(self.multiple_eras_in_sequence):
            start = index * self.max_seq_len % self.data.shape[0]
            end   = np.min([start + self.max_seq_len, self.data.shape[0]])
            # print(f"start: {start}, end: {end}")

            features = pd.DataFrame(self.data[self.feature_names].values[start: end])
            targets  = pd.DataFrame(self.data[self.target_names].values[start: end])
            eras     = pd.Series(self.data[self.era_col_name].values[start: end] - self.data[self.era_col_name].values[start: end].min() + 1)
        else:
            # Most performant filtering method https://medium.com/@thomas-jewson/faster-pandas-what-is-the-most-performant-filtering-method-a5dbb8f694dc#:~:text=Between%200%20and%2010%2C000%2C000%20the,query%20is%20the%20slowest%20method.
            features = self.data[self.feature_names][self.data[self.era_col_name].values == index + self.min_era]
            targets  = self.data[self.target_names][self.data[self.era_col_name].values == index + self.min_era]
            masks    = torch.ones((features_pt.shape[0], 1), dtype=torch.float)#dummy mask, no padded values => all 1s
            eras     = None

        # Features and Targets converted to
        features_pt, targets_pt = self.convert_to_torch(features, data_type=np.int8).float(), self.convert_to_torch(targets, data_type=np.float32)
        eras_pt = self.convert_to_torch(eras, data_type=np.int8) if eras is not None else eras
        eras_pt = eras_pt.unsqueeze(-1).int()

        if(self.add_padding):
            return self.pad_sequence(features_pt, targets_pt, eras_pt)
        else:
            return features_pt, targets_pt, masks, eras_pt

    def pad_sequence(self, features_pt, targets_pt, eras_pt):
        pad_len = self.max_seq_len - features_pt.shape[0]
        padded_features_pt = F.pad(features_pt, (0, 0, 0, pad_len), value=self.padding_value)
        padded_targets_pt  = F.pad(targets_pt, (0, 0, 0, pad_len), value=self.padding_value)

        # The mask indicates when the sequence actually ends
        masks = torch.ones((features_pt.shape[0], 1), dtype=torch.float)
        masks = torch.cat([masks, torch.zeros((pad_len, 1), dtype=torch.float)], dim=0)

        # Adds self.padding_value if needed to eras
        padded_eras_pt = torch.cat([eras_pt, torch.zeros((pad_len, 1), dtype=torch.int)], dim=0)

        return padded_features_pt, padded_targets_pt, masks, padded_eras_pt

    @staticmethod
    def convert_to_torch(data, data_type=np.int8):
        data = torch.from_numpy(
                    data.values.astype(data_type))

        return data

## TODO
# Rajouter toutes les cryptos
# Rajouter les finance news https://medium.com/codex/extracting-financial-news-seamlessly-using-python-4dcc732d9ff1
# OPEN AI Embeddings des news
# PCA des embeddings 
# Faire du weekly prediction
class CryptoDataset(Dataset):
    def __init__(self, start_date, end_date):
        # pip install newsapi-python
        # Download cryptocurrency price data
        yf.pdr_override()
        btc = pdr.get_data_yahoo('BTC-USD', start_date, end_date)
        eth = pdr.get_data_yahoo('ETH-USD', start_date, end_date)
        ltc = pdr.get_data_yahoo('LTC-USD', start_date, end_date)

        # Combine price data into one dataframe
        df = pd.concat([btc['Close'], eth['Close'], ltc['Close']], axis=1)
        df.columns = ['BTC', 'ETH', 'LTC']

        # Add technical analysis features
        df = ta.add_all_ta_features(df, open='BTC', high='BTC', low='BTC', close='BTC', volume='BTC')

        # Download news data
        newsapi = NewsApiClient(api_key='YOUR_API_KEY')
        headlines = newsapi.get_everything(q='cryptocurrency', from_param=start_date, to=end_date, language='en')

        # Combine news data into one dataframe
        articles = []
        for article in headlines['articles']:
            articles.append({'title': article['title'], 'description': article['description'], 'content': article['content']})
        news_df = pd.DataFrame(articles)

        # Combine price and news data into one dataframe
        df = pd.concat([df, news_df], axis=1)

        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get features and target for this index
        features = self.data.iloc[index, :-3]
        target = self.data.iloc[index, -3:]

        # Convert to torch tensors
        features = torch.tensor(features.values, dtype=torch.float32)
        target = torch.tensor(target.values, dtype=torch.float32)

        return features, target

class CryptoMetrics:
    def __init__(self, symbols, interval, start_date, end_date):
        self.symbols = symbols
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_crypto_data()

    def get_crypto_data(self):
        url = "https://api.binance.com/api/v3/klines"
        dfs = []
        for symbol in self.symbols:
            df = pdr.data.get_data_yahoo(symbol, self.start_date, self.end_date, period='1d')
            
            # Do not concatenate if dataframe is empty
            if not df.empty:
                df['Crypto_Symbol'] = symbol.split('-USD')[0]            
                dfs.append(df)
            
        data = pd.concat(dfs, axis=0).sort_index()
        data.index = pd.MultiIndex.from_arrays([data.index.to_list(), data.Crypto_Symbol.to_list()], names=['Date', 'Crypto_Symbol'])

        return data

    def compute_volatility(self):
        log_returns = np.log(self.data['Close'] / self.data.groupby(level=1)['Close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)
        return volatility
    
    def compute_return(self, d):
        return (self.data.groupby(level=1)['Close'].shift(-d) - self.data['Close']) / self.data['Close']

    def compute_first_derivative(self):
        return self.data.groupby(level=1)['Close'].diff()        

    def compute_moving_average(self, window):
        return self.data.groupby(level=1)['Close'].rolling(window=window).mean()

    def compute_rsi(self, window):
        delta = self.compute_first_derivative()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, fast_window, slow_window, signal_window):
        ema_fast = self.data.groupby(level=1)['Close'].ewm(span=fast_window, adjust=False).mean()
        ema_slow = self.data.groupby(level=1)['Close'].ewm(span=slow_window, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def compute_obv(self):
        obv = np.where(self.data['Close'] > self.data.groupby(level=1)['Close'].shift(1), self.data['Volume'], -self.data['Volume'])
        obv = obv.cumsum()
        return obv

    def compute_mfi(self, window):
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        raw_money_flow = typical_price * self.data['Volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi


def get_crypto_news(crypto_symbol, api_key, start_date=datetime.datetime(2023, 10, 1), end_date=datetime.datetime(2023, 10, 2)):
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={crypto_symbol}&from={start_date_str}&to={end_date_str}'
    news_json = requests.get(url).json()
    
    data_dict = {'date':[], 'title':[], 'sentiment':[]}
    
    for el in news_json:
        data_dict['date'].append(el['date'])
        data_dict['title'].append(el['title'])
        data_dict['sentiment'].append(el['sentiment'])
            
    return pd.DataFrame(data_dict)

if __name__ == "__main__":
    ## top 30 Cryptos by volume traded (24h) https://finance.yahoo.com/u/yahoo-finance/watchlists/crypto-top-volume-24hr/
    # Removed TUSD, USDC, USDCE, USDT because they are stable coins
    # Added "BCH-USD", "ALGO-USD", "MANA-USD" instead
    # Added Symbold from binance, the objective is to reach 50 symbols
    symbols = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "TRX-USD", 
    "LINK-USD", "MATIC-USD", "DOT-USD", "WBTC-USD", "LTC-USD", "DAI-USD", "SHIB-USD", "BCH-USD",   
    "AVAX-USD", "XLM-USD", "ATOM-USD", "ETC-USD", "UNI-USD", "FIL-USD", "LDO-USD", "HBAR-USD", "APT-USD",
    "BTCB-USD", "BUSD-USD", "ARB11841-USD", "NEO-USD", "FDUSD-USD", "GALA-USD", "PEPE24478-USD", "ORDI-USD", 
    "STORJ-USD", "GAS-USD", "MEME28301-USD", "HIFI23037-USD", "WETH-USD", "MANA-USD", "ALGO-USD",
    "ICP-USD", "WBET-USD", "VET-USD", "OP-USD", "ARB-USD", "NEAR-USD", "AAVE-USD", "INJ-USD", "MKR-USD", "RUNE-USD",
    "QNT-USD", "GRT-USD", "IMX-USD", "EGLD-USD"
    ]#, "TUSD-USD", "USDC-USD", "USDCE-USD", "USDT-USD"
    
    # Dataset starts in the middle of the first crypto boom, volatility is high.
    start_date = datetime.datetime(2017, 12, 1)
    end_date = datetime.datetime(2023, 12, 1)
    crypto_data = CryptoMetrics(symbols=symbols, interval='1d', start_date=start_date, end_date=end_date)
    (crypto_data.data.Crypto_Symbol.value_counts()>=crypto_data.data.Crypto_Symbol.value_counts().max()).sum()
    crypto_data.data.Crypto_Symbol.describe()
    crypto_data.compute_return(20).groupby(level=1).mean()
    
    crypto_data.data.groupby(level=1).shift(-1)
    
    url = f'https://eodhd.com/api/exchange-symbol-list/US?api_token=654cea806906e1.04970075&fmt=json'
    tickers = requests.get(url).json()
    
    amzn_news = get_crypto_news('DOGE', EODHD_news_api_key, start_date, end_date)
    #Manual scrape: https://github.com/nicknochnack/Stock-and-Crypto-News-ScrapingSummarizationSentiment/blob/main/Scrape%20and%20Summarize%20Stock%20News%20using%20Python%20and%20Deep%20Learning-Tutorial.ipynb
    #Finbert for sentiment analysis: https://huggingface.co/ProsusAI/finbert
    #https://finance.yahoo.com/topic/crypto/
    
    start_date = datetime.datetime(2017, 12, 1)    
    end_date = datetime.datetime(2023, 12, 1)    
    s=start_date.strftime('%Y-%m-%d')
    e=end_date.strftime('%Y-%m-%d')
    symbol = 'BTC'
    url = (f'https://newsapi.org/v2/everything?q={symbol}&'
            f'from={s}&to={e}&'
            f'language=en&'
            f'apiKey={NewsAPI_key}')
        # 'sortBy=popularity&'
        # f'domains=google.com,yahoo.com,finance,wired&'


    response = requests.get(url)
    response_dict = response.json()

    print(response_dict['totalResults'])
    response_dict['articles'][0].keys()#publishedAt is timestamp
    response_dict['articles'][0]['publishedAt']
    # Doc: https://newsapi.org/docs/endpoints/everything
    # Limited by 100 articles per day...
    # Last resort, twitter: https://medium.com/@kccmeky/how-to-collect-crypto-news-on-twitter-using-python-56f31639922e
    # https://github.com/nicknochnack/Stock-and-Crypto-News-ScrapingSummarizationSentiment/blob/main/Scrape%20and%20Summarize%20Stock%20News%20using%20Python%20and%20Deep%20Learning-Tutorial.ipynb
    # https://www.google.com/search?q=yahoo+finance+BTC&sca_esv=581117380&source=lnt&tbs=cdr%3A1%2Ccd_min%3A11%2F1%2F2023%2Ccd_max%3A11%2F10%2F2023&tbm=nws
    # https://www.google.com/search?q=yahoo+finance+BTC&tbm=nws
    # How to specify date in google search!
    # https://community.openai.com/t/embedding-text-length-vs-accuracy/96564/13 for OPENAI embeds


    # dataset = load_dataset("monash_tsf", "tourism_monthly")
    # train_ds, val_ds, test_ds = dataset.values()
    # len(train_ds[0]['target'])
    # len(test_ds[0]["target"])
    # # Add your code here to execute when this script is run as the main script

    # train_example = train_ds[0]
    # validation_example = val_ds[0]
    # test_example = test_ds[0]

    # figure, axes = plt.subplots()
    # axes.plot(train_example["target"], color="blue")
    # axes.plot(validation_example["target"], color="red", alpha=0.5)
    # axes.plot(test_example["target"], color="green", alpha=0.5)

    # plt.savefig("here.png")



    # yf.pdr_override()

    # btc = pdr.data.get_data_yahoo(['BTC-USD', 'ETH-USD', 'LTC-USD'], start_date, end_date)


    # crypto_symbols = yf.Tickers("")  # pass an empty string to get all available symbols
    # crypto_symbols = crypto_symbols.tickers  # get the list of symbols
    # print(crypto_symbols)

