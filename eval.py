import argparse
import os
import torch
from models import E1, E2, Decoder
from utils import save_imgs, load_model_for_eval


def eval(args):
    e1 = E1(args.sep, int((args.resize / 64)))
    e2 = E2(args.sep, int((args.resize / 64)))
    decoder = Decoder(int((args.resize / 64)))

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        decoder = decoder.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model_for_eval(save_file, e1, e2, decoder)

    e1 = e1.eval()
    e2 = e2.eval()
    decoder = decoder.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    save_imgs(args, e1, e2, decoder)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--out', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_display', type=int, default=20)

    args = parser.parse_args()

    eval(args)
