import argparse
import json
import yaml
import os
from yaml.loader import SafeLoader
from googletrans import Translator
from google_drive_downloader import GoogleDriveDownloader as gdd


def download_ds(trans_lang):
    with open('./translations/tr.json') as f:
        url_dict = json.loads(f.read())
    try:
        url = url_dict[trans_lang]
        gdd.download_file_from_google_drive(file_id=url,
                                            dest_path=f'./Massive/translated/data/{trans_lang}-{trans_lang.upper()}.json')
        print(f'{trans_lang}: downloaded.')
        return True
    except KeyError:
        return False


def translations_generator(path, dest_path):
    translator = Translator()
    with open('./translations/translated_langs.yml') as f:
        data = yaml.load(f, Loader=SafeLoader)
    languages = data['lang']
    for trans_lang in languages:
        if not download_ds(trans_lang):
            pth = dest_path + '/' + trans_lang + '-' + trans_lang.upper() + '.jsonl'
            if os.path.exists(pth):
                raise FileExistsError
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            jsons = []
            for idx, line in enumerate(lines):
                print(f'{trans_lang}: {idx + 1} / {len(lines)}')
                js = json.loads(line)
                jsons.append(js)
                translated_txt = translator.translate(js["utt"], dest=trans_lang).text
                js["utt"] = translated_txt
                js["annot_utt"] = translated_txt
                js['locale'] = trans_lang + '-' + trans_lang.upper()
                with open(pth, 'a', encoding='utf8') as f:
                    cont = json.dumps(js, ensure_ascii=False)
                    f.write(cont)
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Translate Massive to Slavic languages.")
    parser.add_argument('-p', '--path', help='path to translated language.')
    parser.add_argument('-dp', '--dst_path', help='translation destination.')
    args = parser.parse_args()
    translations_generator(args.path, args.dst_path)
