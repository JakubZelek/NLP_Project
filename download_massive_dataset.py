import glob
import json
import os
import tarfile
import urllib.request

import torch
from transformers import MT5Model, T5Tokenizer, MT5Config, AutoTokenizer, AutoModelForMaskedLM, \
    AutoModelForSeq2SeqLM

cwd = os.getcwd()


class MASSIVE:
    """Massive dataset."""

    def __init__(self, root: str, language='pl', download: bool = False):
        """Massive dataset init.
        Args:
            root (str): path to directory where dataset can be extracted (if download == True) or where dataset is.
            language (str): language with which we will work.
            Polish can be set by language = 'pl'.
            """

        self.root = root
        if download:
            massive_file = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
            print("DOWNLOADING...")
            urllib.request.urlretrieve(massive_file, self.root + "/massive.tar.gz")
            print("EXTRACTING TO " + root + '/Massive')
            file = tarfile.open(self.root + "/massive.tar.gz")
            file.extractall(root + '/Massive')
            file.close()
            os.remove(self.root + "/massive.tar.gz")
            print("REMOVING massive.tar.gz")
            print("DONE.")

        file_name = glob.glob(root + '/Massive/1.0/data/' + language + '*.jsonl')[0]
        with open(file_name) as f:
            json_list = map(lambda json_str: json.loads(json_str), f.read().split('\n'))

        self.dict = {}
        for json_elem in json_list:
            idx = int(json_elem['id'])
            del json_elem['id']
            self.dict[idx] = json_elem

    def __getitem__(self, index: int):
        """[] operator for MASSIVE class."""
        return self.dict[index]


class DownloadMT5:
    def __init__(self, path):
        config = MT5Config(d_model=768, d_ff=2048, num_layers=12, num_heads=12)
        self.model = MT5Model.from_pretrained("google/mt5-base", ignore_mismatched_sizes=True,
                                              config=config)
        self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
        torch.save(self.model.state_dict(), path + 'mt5-base.bin')
        self.tokenizer.save_vocabulary(save_directory=path)


class DownloadMT5T2T:
    def __init__(self, path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
        torch.save(self.model.state_dict(), path + 'mt5-base-t2t.bin')


class DownloadXLMR:
    def __init__(self, path):
        self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        torch.save(self.model.state_dict(), path + 'xlm-r-base.bin')
        self.tokenizer.save_vocabulary(save_directory=path)


if __name__ == '__main__':
    M = MASSIVE(cwd, download=True)
    DownloadMT5(cwd + '/saved_models/')
    DownloadMT5T2T(cwd + '/saved_models/')
    DownloadXLMR(cwd + '/saved_models/')
