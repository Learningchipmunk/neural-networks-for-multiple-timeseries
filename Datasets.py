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

class NumeraiDataset(Dataset):
    '''Defines the Dataset class for numerai data. Example usage:

        train_dataset_pt = NumeraiData(train, feature_cols, target_names, max_seq_len, PADDING_VALUE, add_padding=True)

    '''
    def __init__(self, dataframe, feature_names, target_names, max_seq_len, padding_value, era_col_name='era_int', multiple_eras_in_sequence=False, hop_len=None, shuffle_intra_era=False, add_padding=True):
        self.data = dataframe
        self.feature_names = feature_names
        self.target_names  = target_names
        self.max_seq_len   = max_seq_len
        self.hop_len       = max_seq_len if hop_len is None else hop_len # useful only if multiple_eras_in_sequence is True
        self.padding_value = padding_value
        self.era_col_name  = era_col_name
        self.multiple_eras_in_sequence = multiple_eras_in_sequence
        self.shuffle_intra_era = shuffle_intra_era
        self.add_padding   = add_padding
        self.min_era       = self.data[era_col_name].values.min()

    def __len__(self):
        if self.multiple_eras_in_sequence:
            return  int(len(self.data) / self.hop_len)+1
        else:    
            return len(self.data.groupby(self.era_col_name))

    def __getitem__(self, index):
        ## For multiple eras:
        # Most performant filtering method https://medium.com/@thomas-jewson/faster-pandas-what-is-the-most-performant-filtering-method-a5dbb8f694dc#:~:text=Between%200%20and%2010%2C000%2C000%20the,query%20is%20the%20slowest%20method.
        if(self.multiple_eras_in_sequence):
            start = index * self.hop_len % self.data.shape[0]
            end   = np.min([start + self.max_seq_len, self.data.shape[0]])
            # print(f"start: {start}, end: {end}")
            data = self.data[start: end]# Data Selected
            eras = pd.Series(data[self.era_col_name].values - data[self.era_col_name].values.min() + 1)
        ## For one era selection:
        else:
            data  = self.data[self.data[self.era_col_name].values == index + self.min_era]# Data Selected
            eras  = np.zeros(data.shape[0])

        # Shuffle Intra Era if Needed
        if(self.shuffle_intra_era):
            data = data.sample(frac=1, replace=False).sort_values(by=self.era_col_name)

        #Split data into features and targets
        features = data[self.feature_names]
        targets  = data[self.target_names]

        # Features and Targets converted to pytorch Tensors
        features_pt, targets_pt = self.convert_to_torch(features, data_type=np.int8).float(), self.convert_to_torch(targets, data_type=np.float32)
        eras_pt = self.convert_to_torch(eras, data_type=np.int8) if eras is not None else eras
        eras_pt = eras_pt.unsqueeze(-1).int()

        # If no padding, the mask is all 1s and is returned as a dummy var
        masks = torch.ones((data.shape[0], 1), dtype=torch.float)#dummy mask, no padded values => all 1s

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
# Rajouter les finance news https://medium.com/codex/extracting-financial-news-seamlessly-using-python-4dcc732d9ff1
# OPEN AI Embeddings des news
# PCA des embeddings 
# Faire du weekly prediction
class CryptoMetrics:
    def __init__(self, symbols, interval, start_date, end_date, get_data=True):
        yf.pdr_override()
        self.symbols = symbols
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        
        if(get_data):
            self.data = self.get_crypto_data()

    def get_crypto_data(self):
        # url = "https://api.binance.com/api/v3/klines"
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

    def compute_volatility(self, window):
        log_returns = np.log(1 + self.data['Close'] / self.data.groupby(level=1)['Close'].shift(1))
        volatility = log_returns.rolling(window=window, min_periods=1).std()
        return volatility
    
    def compute_return(self, d):
        return (self.data.groupby(level=1)['Close'].shift(-d) - self.data['Close']) / self.data['Close']

    def compute_first_derivative(self):
        return self.data.groupby(level=1)['Close'].diff()        

    def compute_moving_average(self, window):
        return self.data.groupby(level=1)['Close'].rolling(window=window).mean()

    def compute_rsi(self, window):
        if 'first_derivative' not in self.data.columns:
            self.data['first_derivative'] = self.compute_first_derivative()
        
        gain = self.data['first_derivative'].where(self.data['first_derivative'] > 0, 0)
        loss = -self.data['first_derivative'].where(self.data['first_derivative'] < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, fast_window=12, slow_window=26, signal_window=9):
        ema_fast = self.data.groupby(level=1)['Close'].ewm(span=fast_window, adjust=False).mean()
        ema_slow = self.data.groupby(level=1)['Close'].ewm(span=slow_window, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return self.restructure_index(macd), self.restructure_index(signal)

    def compute_adx(self, window=14):
        '''https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx#:~:text=The%20Average%20Directional%20Index%20(ADX)%20is%20used%20to%20measure%20the,edge%20when%20%2DDI%20is%20greater.'''
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        tr1 = high - low
        tr2 = abs(high - close.groupby(level=1).shift())
        tr3 = abs(low - close.groupby(level=1).shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr        = true_range.groupby(level=1).rolling(window=window).mean()

        up_move   = high - high.groupby(level=1).shift()
        down_move = low.groupby(level=1).shift() - low

        plus_dm  = pd.DataFrame(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=self.data.index)
        minus_dm = pd.DataFrame(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=self.data.index)
        
        plus_dm_avg  = plus_dm.groupby(level=1).rolling(window=window).mean()
        minus_dm_avg = minus_dm.groupby(level=1).rolling(window=window).mean()

        plus_di  = 100 * (plus_dm_avg.values.squeeze() / atr.values.squeeze())
        minus_di = 100 * (minus_dm_avg.values.squeeze() / atr.values.squeeze())

        dx  = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx  = pd.DataFrame(dx, index=self.data.index)
        adx = dx.groupby(level=1).rolling(window=window).mean()

        return self.restructure_index(adx)

    def compute_obv(self):
        obv = np.where(self.data['Close'] > self.data.groupby(level=1)['Close'].shift(1), self.data['Volume'], -self.data['Volume'])
        obv = obv.cumsum()
        return obv

    def compute_mfi(self, window):
        if not hasattr(self, 'typical_price_'):
            self.typical_price_  = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
            self.raw_money_flow_ = self.typical_price_ * self.data['Volume']
            self.positive_flow_  = pd.DataFrame(np.where(self.typical_price_ > self.typical_price_.groupby(level=1).shift(1), self.raw_money_flow_, 0), index=self.data.index)
            self.negative_flow_  = pd.DataFrame(np.where(self.typical_price_ < self.typical_price_.groupby(level=1).shift(1), self.raw_money_flow_, 0), index=self.data.index)
        positive_flow_sum = self.positive_flow_.groupby(level=1).rolling(window=window).sum()
        negative_flow_sum = self.negative_flow_.groupby(level=1).rolling(window=window).sum()

        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return self.restructure_index(mfi)

    def create_dataset_with_metrics(self, path='CryptoData/CryptoData_WithMetrics.csv'):
        # Adds Features
        feature_labels = ['vol_3']
        self.data[feature_labels[-1]] = self.compute_volatility(window=3)
        window_values = [7, 14, 28]

        for window in window_values:
            feature_labels.append(f'vol_{window}')
            self.data[feature_labels[-1]] = self.compute_volatility(window=window)

            feature_labels.append(f'mfi_{window}')
            self.data[feature_labels[-1]] = self.compute_mfi(window=window)

            feature_labels.append(f'rsi_{window}')
            self.data[feature_labels[-1]] = self.compute_rsi(window=window)

            feature_labels.append(f'adx_{window}')
            self.data[feature_labels[-1]] = self.compute_adx(window=window)
            
        feature_labels += ['macd', 'signal']
        self.data['macd'], self.data['signal'] = self.compute_macd(fast_window=12, slow_window=26, signal_window=9)
            
        # Adds Targets
        target_values = [1, 7, 14, 20, 30, 60]
        target_labels = []
        
        for target in target_values:
            target_labels.append(f'{target}d_return')
            self.data[target_labels[-1]]  = self.compute_return(target)
            
        data = self.data[feature_labels + target_labels]
        tuples = [('Features', feature) for feature in feature_labels] + [('Targets', target) for target in target_labels]
        data.columns = pd.MultiIndex.from_tuples(tuples)
        
        data.to_csv(path)
        return data
    
    @staticmethod
    def restructure_index(df):
        df.index  = df.index.droplevel(0)
        df        = df.reorder_levels(['Date', 'Crypto_Symbol']).sort_index()        
        return df    
    
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
    # API keys for NewsAPI and EODHD
    with open('api_keys.json') as f:
        api_keys = json.load(f)
    EODHD_news_api_key = api_keys['EODHD_news_api_key']
    NewsAPI_key = api_keys['NewsAPI_key']	
    
    ## top 30 Cryptos by volume traded (24h) https://finance.yahoo.com/u/yahoo-finance/watchlists/crypto-top-volume-24hr/
    # Removed TUSD, USDC, USDCE, USDT because they are stable coins
    # Added "BCH-USD", "ALGO-USD", "MANA-USD" instead
    # Added Symbold from binance, the objective is to reach 50 symbols
    symbols = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "TRX-USD", 
    "LINK-USD", "MATIC-USD", "DOT-USD", "WBTC-USD", "LTC-USD", "DAI-USD", "SHIB-USD", "BCH-USD",   
    "AVAX-USD", "XLM-USD", "ATOM-USD", "ETC-USD", "UNI-USD", "FIL-USD", "LDO-USD", "HBAR-USD", "APT-USD",
    "BTCB-USD", "BUSD-USD", "ARB11841-USD", "NEO-USD", "GALA-USD", "PEPE24478-USD", 
    "STORJ-USD", "GAS-USD", "MEME28301-USD", "HIFI23037-USD", "WETH-USD", "MANA-USD", "ALGO-USD",
    "ICP-USD", "VET-USD", "OP-USD", "ARB-USD", "NEAR-USD", "AAVE-USD", "INJ-USD", "MKR-USD", "RUNE-USD",
    "QNT-USD", "GRT-USD", "IMX-USD", "EGLD-USD"
    ]#Removed pegged cryptos, it is not interesting to predict their prices: "TUSD-USD", "USDC-USD", "USDCE-USD", "USDT-USD"
    # Not Found: "FDUSD-USD", "ORDI-USD", "WBET-USD", 
    
    # Dataset starts in the middle of the first crypto boom, volatility is high.
    start_date = datetime.datetime(2017, 12, 1)
    end_date = datetime.datetime(2023, 12, 1)
    crypto_data = CryptoMetrics(symbols=symbols, interval='1d', start_date=start_date, end_date=end_date, get_data=False)
    
    # crypto_symbols = crypto_data.data.Crypto_Symbol.unique().tolist()
    # np.save('CryptoData/CryptoSymbols.npy', crypto_symbols)
    
    # crypto_data.data.to_csv('CryptoData/CryptoData.csv')


    # Loading already downloaded data
    crypto_data.data = pd.read_csv('CryptoData/CryptoData.csv', index_col=[0, 1], parse_dates=True)
    
    
    # (crypto_data.data.Crypto_Symbol.value_counts()>=crypto_data.data.Crypto_Symbol.value_counts().max()).sum()
    # crypto_data.compute_return(20).groupby(level=1).mean()
    
    data = crypto_data.create_dataset_with_metrics(path='CryptoData/CryptoData_WithMetrics.csv')
    data['Targets'].dropna().groupby(level=1).mean()

    # url = f'https://eodhd.com/api/exchange-symbol-list/US?api_token=654cea806906e1.04970075&fmt=json'
    # tickers = requests.get(url).json()
    
    # amzn_news = get_crypto_news('DOGE', EODHD_news_api_key, start_date, end_date)
    # #Manual scrape: https://github.com/nicknochnack/Stock-and-Crypto-News-ScrapingSummarizationSentiment/blob/main/Scrape%20and%20Summarize%20Stock%20News%20using%20Python%20and%20Deep%20Learning-Tutorial.ipynb
    # #Finbert for sentiment analysis: https://huggingface.co/ProsusAI/finbert
    # #https://finance.yahoo.com/topic/crypto/
    
    # start_date = datetime.datetime(2017, 12, 1)    
    # end_date = datetime.datetime(2023, 12, 1)    
    # s=start_date.strftime('%Y-%m-%d')
    # e=end_date.strftime('%Y-%m-%d')
    # symbol = 'BTC'
    # url = (f'https://newsapi.org/v2/everything?q={symbol}&'
    #         f'from={s}&to={e}&'
    #         f'language=en&'
    #         f'apiKey={NewsAPI_key}')
    #     # 'sortBy=popularity&'
    #     # f'domains=google.com,yahoo.com,finance,wired&'


    # response = requests.get(url)
    # response_dict = response.json()

    # print(response_dict['totalResults'])
    # response_dict['articles'][0].keys()#publishedAt is timestamp
    # response_dict['articles'][0]['publishedAt']
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

