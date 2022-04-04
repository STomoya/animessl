
import argparse
import os
from typing import Any
import subprocess

'''pretrained models info'''
'''utils'''

class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    def __delattr__(self, name: str) -> None:
        del self[name]

def _add_model(name, id, filename=None):
    model_info = EasyDict()
    model_info.NAME            = name
    model_info.GOOGLE_DRIVE_ID = id
    if filename is None:
        filename = name + '.pth'
    model_info.OUTPUT_FILENAME = filename
    return model_info

'''info'''

PRETRAINED = EasyDict()

PRETRAINED.OUTPUT_FOLDER = './weights'

PRETRAINED.MODELS = EasyDict()
PRETRAINED.MODELS.SIMCLR_RESNET50 = _add_model('simclr_resnet50', '1bGKrxM_ciCgMKsrRfKQ1lDsx84Xc_tON')
PRETRAINED.MODELS.SWAV_RESNET50   = _add_model('swav_resnet50',   '1rDpfwc7gLUb2BQEJ_oWygVph2tszh-xk')
PRETRAINED.MODELS.SIMCLR_CONVNEXT = _add_model('simclr_convnext', '1hMO_63Sz0ZIeSR6I9onmU71aYDNxZWXK')

all_models = [key.lower() for key in PRETRAINED.MODELS.keys()]

''''''

def download(id, filename, shell_script='./scripts/download_from_google_drive_wget.sh'):
    '''download file from google drive via wget/curl.
    See `./scripts/download_from_google_drive_wget.sh` or `./scripts/download_from_google_drive_curl.sh`
    NOTE: Vissl docker image only installs wget. Add curl installation in Dockerfile to use curl.
    '''
    command = f'/bin/bash {shell_script} {id} {filename}'
    subprocess.run(command, shell=True)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=all_models)
    parser.add_argument('--out-folder')
    return parser.parse_args()

def main():
    args = get_args()
    if args.out_folder:
        PRETRAINED.OUTPUT_FOLDER = args.out_folder
    if not os.path.exists(PRETRAINED.OUTPUT_FOLDER):
        os.makedirs(PRETRAINED.OUTPUT_FOLDER)

    model_info = PRETRAINED.MODELS[args.name.upper()]
    download(model_info.GOOGLE_DRIVE_ID, os.path.join(PRETRAINED.OUTPUT_FOLDER, model_info.OUTPUT_FILENAME))

if __name__=='__main__':
    main()
