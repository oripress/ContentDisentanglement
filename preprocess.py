import os
import argparse


#######
# CelebA attributes:
# ------
# 5_o_Clock_Shadow 1
# Arched_Eyebrows 2
# Attractive 3
# Bags_Under_Eyes 4
# Bald 5
# Bangs 6
# Big_Lips 7
# Big_Nose 8
# Black_Hair 9
# Blond_Hair 10
# Blurry 11
# Brown_Hair 12
# Bushy_Eyebrows 13
# Chubby 14
# Double_Chin 15
# Eyeglasses 16
# Goatee 17
# Gray_Hair 18
# Heavy_Makeup 19
# High_Cheekbones 20
# Male 21
# Mouth_Slightly_Open 22
# Mustache 23
# Narrow_Eyes 24
# No_Beard 25
# Oval_Face 26
# Pale_Skin 27
# Pointy_Nose 28
# Receding_Hairline 29
# Rosy_Cheeks 30
# Sideburns 31
# Smiling 32
# Straight_Hair 33
# Wavy_Hair 34
# Wearing_Earrings 35
# Wearing_Hat 36
# Wearing_Lipstick 37
# Wearing_Necklace 38
# Wearing_Necktie 39
# Young 40
#######

def preprocess_celeba(args):
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    allA = []
    allB = []

    with open(args.attributes) as f:
        lines = f.readlines()

    # Eyeglasses
    if args.config == 'glasses':
        for line in lines[2:]:
            line = line.split()
            if int(line[16]) == 1:
                allA.append(line[0])
            elif int(line[16]) == -1:
                allB.append(line[0])

    # Mouth slightly open
    if args.config == 'mouth':
        for line in lines[2:]:
            line = line.split()
            if int(line[22]) == 1:
                allA.append(line[0])
            else:
                allB.append(line[0])

    # Facial hair
    if args.config == 'beard':
        for line in lines[2:]:
            line = line.split()
            if int(line[21]) == 1 and int(line[1]) == -1 and (int(line[23]) == 1 or int(line[17]) == 1 or int(
                    line[25]) == -1):  # male AND (mustache OR goatee OR beard) AND (no shadow)
                allA.append(line[0])
            elif int(line[21]) == 1 and int(line[25]) == 1 and int(line[1]) == -1:  # male AND (no beard, no shadow)
                allB.append(line[0])

    # Custom
    if args.config == 'custom':
        for line in lines[2:]:
            line = line.split()
            if int(line[args.custom]) == 1:
                allA.append(line[0])
            else:
                allB.append(line[0])

    testA = allA[:args.num_test_imgs]
    testB = allB[:args.num_test_imgs]
    trainA = allA[args.num_test_imgs:]
    trainB = allB[args.num_test_imgs:]

    with open(os.path.join(args.dest, 'testA.txt'), 'w') as f:
        for i, _img in enumerate(testA):
            if i == len(testA) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'testB.txt'), 'w') as f:
        for i, _img in enumerate(testB):
            if i == len(testB) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'trainA.txt'), 'w') as f:
        for i, _img in enumerate(trainA):
            if i == len(trainA) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'trainB.txt'), 'w') as f:
        for i, _img in enumerate(trainB):
            if i == len(trainB) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))


def preprocess_folders(args):
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    trainA = os.listdir(os.path.join(args.root), 'trainA')
    trainB = os.listdir(os.path.join(args.root), 'trainB')
    testA = os.listdir(os.path.join(args.root), 'testA')
    testB = os.listdir(os.path.join(args.root), 'testB')

    with open(os.path.join(args.dest, 'testA.txt'), 'w') as f:
        for i, _img in enumerate(testA):
            if i == len(testA) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'testB.txt'), 'w') as f:
        for i, _img in enumerate(testB):
            if i == len(testB) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'trainA.txt'), 'w') as f:
        for i, _img in enumerate(trainA):
            if i == len(trainA) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))

    with open(os.path.join(args.dest, 'trainB.txt'), 'w') as f:
        for i, _img in enumerate(trainB):
            if i == len(trainB) - 1:
                f.write("%s" % os.path.join(args.root, _img))
            else:
                f.write("%s\n" % os.path.join(args.root, _img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="",
                        help="path to the celeba folder, or if you\'re using another dataset this should be the path to the root")
    parser.add_argument("--dest", default="", help="path to the destination folder")
    parser.add_argument("--attributes", default="", help="path to the attributes file")
    parser.add_argument("--num_test_imgs", default=64, help="number of images in the test set")
    parser.add_argument("--config", default="glasses", help="configs available: glasses, mouth, beard")
    parser.add_argument("--custom", default=32, help="use a custom celeba attribute")
    parser.add_argument("--folders", action="store_true",
                        help="use custom folders, instead of celeba")

    args = parser.parse_args()

    if not args.folders:
        preprocess_celeba(args)
    else:
        preprocess_folders(args)
