import urllib.request
import tarfile
import os
import glob
import json


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
        print(file_name)
        with open(file_name) as f:
            json_list = map(lambda json_str: json.loads(json_str), f.read().split('\n'))

        self.dict = {}
        for json_elem in json_list:
            idx = int(json_elem['id'])
            del json_elem['id']
            self.dict[idx] = json_elem

    def __getitem__(self, index: int):
        """[] operator for MASSIVE class."""
        # we'll see what getting will look like, because maybe we don't need all the information.
        return self.dict[index]


if __name__ == '__main__':
    cwd = os.getcwd()
    M = MASSIVE(cwd, download=False)
    print(M[1])
