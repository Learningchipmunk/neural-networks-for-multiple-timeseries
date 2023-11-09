
import os
import torch
# import ta
import requests
import json
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import yfinance as yf

import torch.nn.functional as F
from torch.utils.data import Dataset
from numerapi import NumerAPI
from datasets import load_dataset


EODHD_news_api_key = '654cea806906e1.04970075' 

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
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.data = self.get_binance_data()

    def get_binance_data(self):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval
        }
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def compute_volatility(self):
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)
        return volatility
    
    def compute_return(self, d):
        return (self.data['close'].shift(d) - self.data['close']) / self.data['close']

    def compute_first_derivative(self):
        return self.data['close'].diff()        

    def compute_moving_average(self, window):
        return self.data['close'].rolling(window=window).mean()

    def compute_rsi(self, window):
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, fast_window, slow_window, signal_window):
        ema_fast = self.data['close'].ewm(span=fast_window, adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=slow_window, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def compute_obv(self):
        obv = np.where(self.data['close'] > self.data['close'].shift(1), self.data['volume'], -self.data['volume'])
        obv = obv.cumsum()
        return obv

    def compute_mfi(self, window):
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        raw_money_flow = typical_price * self.data['volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi



if __name__ == "__main__":

    dataset = load_dataset("monash_tsf", "tourism_monthly")
    train_ds, val_ds, test_ds = dataset.values()
    len(train_ds[0]['target'])
    len(test_ds[0]["target"])
    # Add your code here to execute when this script is run as the main script

    import matplotlib.pyplot as plt

    train_example = train_ds[0]
    validation_example = val_ds[0]
    test_example = test_ds[0]

    figure, axes = plt.subplots()
    axes.plot(train_example["target"], color="blue")
    axes.plot(validation_example["target"], color="red", alpha=0.5)
    axes.plot(test_example["target"], color="green", alpha=0.5)

    plt.savefig("here.png")



    start_date = datetime.datetime(2018, 9, 10)
    end_date = datetime.datetime(2022, 9, 10)
    yf.pdr_override()

    btc = pdr.data.get_data_yahoo(['BTC-USD'], start_date, end_date)


