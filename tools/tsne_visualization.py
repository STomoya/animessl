
import sys
import argparse
import logging
import os, glob
import re
logging.basicConfig(format='%(levelname)s:%(asctime)s:%(filename)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# visualization
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm

# loading images
from PIL import Image
import torchvision.transforms.functional as TF

# loading model
from inference_model import get_trained_model
import torch
import torch.nn as nn
import torchvision.models as models

def process_image(path):
    image = Image.open(path).convert('RGB')
    image = TF.resize(image, (224, 224))
    image = TF.to_tensor(image)
    image = TF.normalize(image,
        [0.7106, 0.6574, 0.6511],
        [0.2561, 0.2617, 0.2539])
    return image.unsqueeze(0)

def prepair_model(args):
    '''prepair model'''
    # if weights are converted to torchvision
    if args.torchvision:
        assert args.weights is not None and os.path.exists(args.weights), f'existing "--weights" required.'
        resnet50 = models.resnet50(pretrained=False)
        resnet50.fc = nn.Identity() # replace last fc
        resnet50.eval()
        state_dict = torch.load(args.weights)
        resnet50.load_state_dict(state_dict)
    # if not, build model from config
    else:
        # auto find required files from checkpoint folder
        if args.train_config is None and args.weights is None:
            assert args.checkpoint_folder, 'either provide "--checkpoint-folder" OR "--weights" and "--train-config"'
            args.train_config = os.path.join(args.checkpoint_folder, 'train_config.yaml')
            assert os.path.exists(args.train_config), f'no train_config.yaml inside "{args.checkpoint_folder}"'
            # all checkpoint files in folder
            temp_ckpt_files = [path for path in glob.glob(os.path.join(args.checkpoint_folder, '*.torch')) if 'phase' in path]
            # find latest by phase number
            number = re.compile('[0-9]+')
            phase_int = [int(number.search(path).group(0)) for path in temp_ckpt_files]
            latest_ckpt = temp_ckpt_files[phase_int.index(max(phase_int))]
            args.weights = latest_ckpt
        # if only one of train-config or weights are provided, abort.
        elif args.train_config is None or args.weights is None:
            logger.warning('Both "--weights" and "--train-config" is required.')
            logger.warning('Aborting.')
            sys.exit(1)
        logger.info(f'[Files for building the model] train_config : {args.train_config}, weights : {args.weights}')
        resnet50 = get_trained_model(
            args.train_config, args.weights)
    return resnet50

def extract(args):
    logger.info('Loading model...')
    model = prepair_model(args)
    logger.info('Succefully loaded model.')

    test_files = np.load(args.file_list)
    if args.limit < len(test_files):
        test_files = test_files[:args.limit]

    logger.info(f'Starting feature extraction on {len(test_files)} files.')
    features = []
    for file in tqdm(test_files, desc='Feature Extraction'):
        feature = model(process_image(file))[0].squeeze(0)
        features.append(feature.detach().cpu().numpy())
    features = np.array(features)
    logger.info('Feature extraction complete.')
    return test_files, features

def visualize(args, files, features):
    logger.info('PCA...')
    PCA_features = PCA(n_components=args.pca_components).fit_transform(features)
    logger.info('PCA complete.')
    logger.info('tSNE...')
    tSNE_features = TSNE().fit_transform(PCA_features)
    logger.info('tSNE complete.')
    logger.info('Creating scatter plot of images...')
    plt.figure(figsize=(60, 40))
    ax = plt.axes()
    bar = tqdm(total=len(tSNE_features), desc='Adding Images')
    for file, feature in zip(files, tSNE_features):
        x, y = feature
        image = OffsetImage(plt.imread(file), zoom=0.2)
        ax.add_artist(AnnotationBbox(image, (x, y), frameon=False))
        bar.update(1)
    ax.scatter(tSNE_features[:, 0], tSNE_features[:, 1])
    ax.autoscale()
    plt.tight_layout()
    plt.savefig(args.output_name)
    logger.info(f'Done. Result is saved as "{args.output_name}.png".')
    logger.info('Closing.')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--torchvision', default=False, action='store_true',
        help='use torchvision.models. weights must be converted using "extra_scripts/convert_vissl_to_torchvision.py"')
    parser.add_argument(
        '--weights', default=None,
        help='path to the weights. required when --torchvision.')
    parser.add_argument(
        '--train-config', default=None,
        help='training config for building the model.')
    parser.add_argument(
        '--checkpoint-folder', default=None,
        help='folder to the checkpoints. if provided, automatically find latest checkpoint and train-config.')
    parser.add_argument(
        '--limit', default=2000, type=int,
        help='number of images to visualise.')
    parser.add_argument(
        '--file-list', default=None, required=True,
        help='path to .npy file with list of paths to images.')
    parser.add_argument(
        '--pca-components', default=50, type=int,
        help='PCA components.')
    parser.add_argument(
        '--output-name', default='scatter-plot',
        help='visualization output filename.')
    args = parser.parse_args()

    files, features = extract(args)
    visualize(args, files, features)
