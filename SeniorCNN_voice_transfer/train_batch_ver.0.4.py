# Change Batch Filename

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import *
from model import *
import time
import math

#J - for batch processing

import json
import base64
import io
import os

# cuda = True if torch.cuda.is_available() else False

cuda = torch.device("cuda:0")


basepath = "input/"
Contentpath = basepath + 'Contents/'
Stylepath = basepath + 'Style/'
Respath = "output/"


# scandirs(path) 폴더 내 특정 확장자 파일 삭제 함수 (txt 파일을 확장자로 가지는 파일 모두 삭제)

def scandirs(path):
    for root, dirs, files in os.walk(path):
        for currentFile in files:
            print("processing file: " + currentFile)
            exts = ('txt')
            if currentFile.lower().endswith(exts):
                os.remove(os.path.join(root, currentFile))

scandirs(Contentpath)     #함수 호출
scandirs(Stylepath)
scandirs(Respath)   #Res 폴더에도 txt 파일 있으면 삭제

# J - Contents file list generation

fContents = open(Contentpath+'conList.txt', 'w', encoding="UTF-8")

for root, dirs, files in os.walk(Contentpath):   # 리스트 파일로 생성할 파일들이 있는 폴더 지정
    files.sort()

    for fname in files:
        full_fname = os.path.join(root, fname)

        if full_fname.endswith('.wav'):     # wav 파일만 생성될 리스트에 기재함
            fContents.write(full_fname)
            fContents.write('\n')

fContents.close()



# J - Style file list generation

fStyle = open(Stylepath+'styleList.txt', 'w', encoding="UTF-8")

for root, dirs, files in os.walk(Stylepath):   # 리스트 파일로 생성할 파일들이 있는 폴더 지정
    files.sort()

    for fname in files:
        full_fname = os.path.join(root, fname)

        if full_fname.endswith('.wav'):     # wav 파일만 생성될 리스트에 기재함
            fStyle.write(full_fname)
            fStyle.write('\n')

fStyle.close()

# 191104_listRes 쓰기모드로 하는 코드

ResName = []

listCon = open(Contentpath+'conList.txt', 'r', encoding="UTF-8")
listSty = open(Stylepath+'styleList.txt', 'r', encoding="UTF-8")

line_num = 1
line_Con = listCon.readline().strip()
line_Sty = listSty.readline().strip()
#line_Res = listRes.readline().strip()

while line_Con and line_Sty:
    CONTENT_FILENAME = line_Con

    STYLE_FILENAME = line_Sty
    ResName.append(CONTENT_FILENAME[15:19] + '+' + STYLE_FILENAME[12:]+'\n')

    line_Con = listCon.readline().strip()
    line_Sty = listSty.readline().strip()

listRes = open(Respath+'listRes.txt', 'w', encoding="UTF-8")
listRes.writelines(ResName)

listCon.close()
listSty.close()
listRes.close()

# wav file reading start
listCon = open(Contentpath+'conList.txt', 'r', encoding="UTF-8")
listSty = open(Stylepath+'styleList.txt', 'r', encoding="UTF-8")
listRes = open(Respath+'listRes.txt', 'r', encoding="UTF-8")

line_num = 1
line_Con = listCon.readline().strip()
line_Sty = listSty.readline().strip()
line_Res = listRes.readline().strip()

while line_Con and line_Sty:
    CONTENT_FILENAME = line_Con
    STYLE_FILENAME = line_Sty

    #
    # listRes.writelines(ResName)

    a_content, sr = wav2spectrum(CONTENT_FILENAME)
    a_style, sr = wav2spectrum(STYLE_FILENAME)

    a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
    if cuda:
        a_content_torch = a_content_torch.cuda()
    print(a_content_torch.shape)
    a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
    if cuda:
        a_style_torch = a_style_torch.cuda()
    print(a_style_torch.shape)

    model = RandomCNN()
    model.eval()

    a_C_var = Variable(a_content_torch, requires_grad=False).float()
    a_S_var = Variable(a_style_torch, requires_grad=False).float()
    if cuda:
        model = model.cuda()
        a_C_var = a_C_var.cuda()
        a_S_var = a_S_var.cuda()

    a_C = model(a_C_var)
    a_S = model(a_S_var)

    # Optimizer
    learning_rate = 0.002
    a_G_var = Variable(torch.randn(a_content_torch.shape).cuda() * 1e-3, requires_grad=True)
    optimizer = torch.optim.Adam([a_G_var])

    # coefficient of content and style
    # style_param = 1
    # J - 2 times style_params
    style_param = 2
    # content_param = 1e2
    # J
    content_param = 1e4

    # num_epochs = 20000
    # print_every = 1000
    # plot_every = 1000

    # J - half of epochs
    num_epochs = 10000
    print_every = 1000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    start = time.time()
    # Train the Model
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        a_G = model(a_G_var)

        content_loss = content_param * compute_content_loss(a_C, a_G)
        style_loss = style_param * compute_layer_style_loss(a_S, a_G)
        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()

        # print
        if epoch % print_every == 0:
            print("{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                          epoch / num_epochs * 100,
                                                                                          timeSince(start),
                                                                                          content_loss.item(),
                                                                                          style_loss.item(),
                                                                                          loss.item()))
            current_loss += loss.item()

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    gen_spectrum = a_G_var.cpu().data.numpy().squeeze()

    #J - Result Path
    gen_audio_C = Respath + line_Res
    spectrum2wav(gen_spectrum, sr, gen_audio_C)

    plt.figure()
    plt.plot(all_losses)

    # plt.savefig('loss_curve.png')
    # J
    plt.savefig(line_Res+'loss_curve.png')

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Content Spectrum")
    # plt.imsave('Content_Spectrum.png', a_content[:400, :])
    # J
    plt.imsave(line_Con+'_Spectrum.png', a_content[:400, :])

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Style Spectrum")

    # plt.imsave('Style_Spectrum.png', a_style[:400, :])
    # J
    plt.imsave(line_Sty+'_Spectrum.png', a_style[:400, :])

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("CNN Voice Transfer Result")
    # plt.imsave('Gen_Spectrum.png', gen_spectrum[:400, :])

    # J
    plt.imsave(Respath+line_Res+'Gen_Spectrum.png', gen_spectrum[:400, :])

    # J - training again: CONTENT_FILENAME 변경 -> 첫번째 CNN 돌린 결과

    CONTENT_FILENAME=Respath + line_Res

    a_content, sr = wav2spectrum(CONTENT_FILENAME)
    a_style, sr = wav2spectrum(STYLE_FILENAME)

    a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
    if cuda:
        a_content_torch = a_content_torch.cuda()
    print(a_content_torch.shape)
    a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
    if cuda:
        a_style_torch = a_style_torch.cuda()
    print(a_style_torch.shape)

    model = RandomCNN()
    model.eval()

    a_C_var = Variable(a_content_torch, requires_grad=False).float()
    a_S_var = Variable(a_style_torch, requires_grad=False).float()
    if cuda:
        model = model.cuda()
        a_C_var = a_C_var.cuda()
        a_S_var = a_S_var.cuda()

    a_C = model(a_C_var)
    a_S = model(a_S_var)

    # Optimizer
    learning_rate = 0.002
    a_G_var = Variable(torch.randn(a_content_torch.shape).cuda() * 1e-3, requires_grad=True)
    optimizer = torch.optim.Adam([a_G_var])

    # coefficient of content and style
    # same as the original data
    style_param = 1
    content_param = 1e2

    num_epochs = 20000
    print_every = 1000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []


    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    start = time.time()
    # Train the Model
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        a_G = model(a_G_var)
        # J - 3 layers Random CNN - err
        # a_G = model(model(a_G_var))

        content_loss = content_param * compute_content_loss(a_C, a_G)
        style_loss = style_param * compute_layer_style_loss(a_S, a_G)
        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()

        # print
        if epoch % print_every == 0:
            print("{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                          epoch / num_epochs * 100,
                                                                                          timeSince(start),
                                                                                          content_loss.item(),
                                                                                          style_loss.item(),
                                                                                          loss.item()))
            current_loss += loss.item()

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    gen_spectrum = a_G_var.cpu().data.numpy().squeeze()

    gen_audio_C =  Respath + '4th'+ line_Res
    spectrum2wav(gen_spectrum, sr, gen_audio_C)

    plt.figure()
    plt.plot(all_losses)

    # plt.savefig('loss_curve.png')
    # J
    plt.savefig(line_Res + '4th'+'loss_curve.png')

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Content Spectrum")
    # plt.imsave('Content_Spectrum.png', a_content[:400, :])
    # J
    plt.imsave(line_Con +'4th'+ '_Spectrum.png', a_content[:400, :])

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("Style Spectrum")

    # plt.imsave('Style_Spectrum.png', a_style[:400, :])
    # J
    plt.imsave(line_Sty + '4th'+'_Spectrum.png', a_style[:400, :])

    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("CNN Voice Transfer Result")
    # plt.imsave('Gen_Spectrum.png', gen_spectrum[:400, :])

    # J
    plt.imsave(Respath + '4th'+line_Res + 'Gen_Spectrum.png', gen_spectrum[:400, :])


    line_Con = listCon.readline().strip()
    line_Sty = listSty.readline().strip()
    line_Res = listRes.readline().strip()

listCon.close()
listSty.close()
listRes.close()



# CONTENT_FILENAME = Contentpath+ "Contents.wav"
# STYLE_FILENAME = basepath + "Style.wav"


