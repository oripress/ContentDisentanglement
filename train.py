import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import E1, E2, Decoder, Disc
from utils import save_imgs, save_model, load_model
from utils import CustomDataset

import argparse


def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    _iter = 0

    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_train = CustomDataset(os.path.join(args.root, 'trainA.txt'), transform=comp_transform)
    domB_train = CustomDataset(os.path.join(args.root, 'trainB.txt'), transform=comp_transform)

    A_label = torch.full((args.bs,), 1)
    B_label = torch.full((args.bs,), 0)
    B_separate = torch.full((args.bs, args.sep * (args.resize // 64) * (args.resize // 64)), 0)

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    decoder = Decoder(args.resize // 64)
    disc = Disc(args.sep, args.resize // 64)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        decoder = decoder.cuda()
        disc = disc.cuda()

        A_label = A_label.cuda()
        B_label = B_label.cuda()
        B_separate = B_separate.cuda()

        mse = mse.cuda()
        bce = bce.cuda()

    ae_params = list(e1.parameters()) + list(e2.parameters()) + list(decoder.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=args.lr, betas=(0.5, 0.999))

    disc_params = disc.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=args.disclr, betas=(0.5, 0.999))

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = load_model(save_file, e1, e2, decoder, ae_optimizer, disc, disc_optimizer)

    e1 = e1.train()
    e2 = e2.train()
    decoder = decoder.train()
    disc = disc.train()

    print('Started training...')
    while True:
        domA_loader = torch.utils.data.DataLoader(domA_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        domB_loader = torch.utils.data.DataLoader(domB_train, batch_size=args.bs,
                                                  shuffle=True, num_workers=6)
        if _iter >= args.iters:
            break

        for domA_img, domB_img in zip(domA_loader, domB_loader):
            if domA_img.size(0) != args.bs or domB_img.size(0) != args.bs:
                break

            domA_img = Variable(domA_img)
            domB_img = Variable(domB_img)

            if torch.cuda.is_available():
                domA_img = domA_img.cuda()
                domB_img = domB_img.cuda()

            domA_img = domA_img.view((-1, 3, args.resize, args.resize))
            domB_img = domB_img.view((-1, 3, args.resize, args.resize))

            ae_optimizer.zero_grad()

            A_common = e1(domA_img)
            A_separate = e2(domA_img)
            A_encoding = torch.cat([A_common, A_separate], dim=1)

            B_common = e1(domB_img)
            B_encoding = torch.cat([B_common, B_separate], dim=1)

            A_decoding = decoder(A_encoding)
            B_decoding = decoder(B_encoding)

            loss = mse(A_decoding, domA_img) + mse(B_decoding, domB_img)

            if args.discweight > 0:
                preds_A = disc(A_common)
                preds_B = disc(B_common)
                loss += args.discweight * (bce(preds_A, B_label) + bce(preds_B, B_label))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5)
            ae_optimizer.step()

            if args.discweight > 0:
                disc_optimizer.zero_grad()

                A_common = e1(domA_img)
                B_common = e1(domB_img)

                disc_A = disc(A_common)
                disc_B = disc(B_common)

                loss = bce(disc_A, A_label) + bce(disc_B, B_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_optimizer.step()

            if _iter % args.progress_iter == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, _iter))

            if _iter % args.display_iter == 0:
                e1 = e1.eval()
                e2 = e2.eval()
                decoder = decoder.eval()

                save_imgs(args, e1, e2, decoder, _iter)

                e1 = e1.train()
                e2 = e2.train()
                decoder = decoder.train()

            if _iter % args.save_iter == 0:
                save_file = os.path.join(args.out, 'checkpoint')
                save_model(save_file, e1, e2, decoder, ae_optimizer, disc, disc_optimizer, _iter)

            _iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--iters', type=int, default=1250000)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--discweight', type=float, default=0.001)
    parser.add_argument('--disclr', type=float, default=0.0002)
    parser.add_argument('--progress_iter', type=int, default=100)
    parser.add_argument('--display_iter', type=int, default=5000)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--load', default='')
    parser.add_argument('--num_display', type=int, default=12)

    args = parser.parse_args()

    train(args)
