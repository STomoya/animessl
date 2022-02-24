
import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-folder', default='.')
    parser.add_argument('--filename')
    return parser.parse_args()

def main():
    args = get_args()
    assert os.path.exists(args.checkpoint), 'File does not exist.'
    if not os.path.join(args.out_folder):
        os.makedirs(args.out_folder)
    if args.filename is None:
        args.filename = os.path.basename(args.checkpoint)

    state_dict = torch.load(args.checkpoint, map_location='cpu')

    if "classy_state_dict" in state_dict.keys():
        model_trunk = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in state_dict.keys():
        model_trunk = state_dict["model_state_dict"]
    else:
        model_trunk = state_dict

    torch.save(model_trunk, os.path.join(args.out_folder, args.filename))

if __name__=='__main__':
    main()
