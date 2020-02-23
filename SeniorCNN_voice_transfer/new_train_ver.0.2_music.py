import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import *
from model import *
import time
import math
cuda = True if torch.cuda.is_available() else False


basepath = "input/"

CONTENT_FILENAME = basepath + "nightcall.wav"
STYLE_FILENAME = basepath + "stairway.wav"

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
    # J - 3 layers Random CNN - err
    #a_G = model(model(a_G_var))

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
                                                                                      style_loss.item(), loss.item()))
        current_loss += loss.item()

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = basepath +"nightcall+stairway.wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)

plt.figure()
plt.plot(all_losses)
plt.savefig('nightcall+stairway_loss_curve.png')

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Content Spectrum")
plt.imsave('nightcall+stairway_Content_Spectrum.png', a_content[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Style Spectrum")
plt.imsave('nightcall+stairway_Style_Spectrum.png', a_style[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("CNN Voice Transfer Result")
plt.imsave('nightcall+stairway_Gen_Spectrum.png', gen_spectrum[:400, :])


# J training again

CONTENT_FILENAME = basepath + "nightcall+stairway.wav"
#J - same style name
# STYLE_FILENAME = basepath + "mv13_t09_s02.wav"

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
    #a_G = model(model(a_G_var))

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
                                                                                      style_loss.item(), loss.item()))
        current_loss += loss.item()

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


gen_spectrum = a_G_var.cpu().data.numpy().squeeze()

gen_audio_C = basepath +"4th_nightcall+stairway.wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)

plt.figure()
plt.plot(all_losses)
plt.savefig('4th loss_curve_nightcall+stairway.png')

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title(basepath +"4th nightcall+stairway_Content Spectrum")
plt.imsave(basepath +'4th nightcall+stairway_Content_Spectrum.png', a_content[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title(basepath +"4th Style Spectrum")
plt.imsave(basepath +'4th nightcall+stairway_Style_Spectrum.png', a_style[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title(basepath +"4th CNN Voice Transfer Result")
plt.imsave(basepath +'4th nightcall+stairway_Gen_Spectrum.png', gen_spectrum[:400, :])

