# coding: UTF-8
import numpy as np
from vae import vae_model
import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def set_model(path):
    if (type(path) == bytes):
        path = path.decode("utf-8")
    global model
    model = vae_model.BetaVAE_B(z_dim=2,nc=1).to(device)
    model.train()
    model.load_state_dict((torch.load(path,map_location="cuda" if cuda else "cpu")))
    model.eval()

def show_result(img, heat_map,save_name=None):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(121)
    plt.imshow(img[:, :], cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(heat_map[:, :], cmap='rainbow')
    plt.axis('off')
    # plt.clim(0,1)
    plt.colorbar()

    if save_name==None:
        plt.show()
        plt.clf()
        plt.close()
    else:
        plt.savefig(save_name)
        plt.clf()


def evaluate_img(img, cut_size, **kwargs):
    height = cut_size
    width = cut_size
    move = cut_size//2
    imgs = [img]
    for key in kwargs:
        imgs.append(kwargs[key])

    sub_img = []
    img_shapes = []
    for im in imgs:
        img_shapes.append(im.shape)
        for i in range(int((im.shape[0] - height) / move) + 1):
            for j in range(int((im.shape[1] - width) / move) + 1):
                sub_img.append(im[i * move:i * move + height, j * move:j * move + width])

    sub_img = np.array(sub_img)
    sub_img = sub_img.reshape(sub_img.shape[0], 1, sub_img.shape[1], sub_img.shape[2])
    data = torch.from_numpy(sub_img).to(device)
    data = torch.where((data < 0), torch.zeros(data.shape).to(device), data)
    data = torch.where((data > 1), torch.ones(data.shape).to(device), data)

    with torch.no_grad():
        # anomaly = model(data, True)
        latents = model(data, True)
        anomaly = torch.norm(latents,dim=1)
        from matplotlib import pyplot as plt
        latents = latents.cpu().numpy()
        fig, ax = plt.subplots()
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.set_aspect(1)
        ax.plot(latents[:,0],latents[:,1], "o")
        plt.show()

    heat_map = np.zeros((img.shape[0],img.shape[1]))
    last_idx = 0

    for channel,img_shape in enumerate(img_shapes):
        h = img.shape[0] // img_shape[0]
        w = img.shape[1] // img_shape[1]
        for i in range(int((img_shape[0] - height) / move) + 1):
            for j in range(int((img_shape[1] - width) / move) + 1):
                idx = i * (int((img_shape[1] - width) / move) + 1) + j + last_idx
                heat_map[h * i * move:h * (i * move + height), w * j * move:w * (j * move + width)] += anomaly[
                    idx].item()
        last_idx = idx
    heat_map = heat_map / (4*len(imgs))
    # heat_map = heat_map / np.max(heat_map)
    print(np.max(heat_map))

    return heat_map


def labview_set_model(num, model_dir="./models", mode="gai"):
    if (type(model_dir) == bytes):
        model_dir = model_dir.decode("utf-8")
    if (type(mode) == bytes):
        mode = mode.decode("utf-8")
    path = model_dir + "/" + str(num) + "/" + mode + "_triplet_weights"
    try:
        set_model(path=path)
    except:
        path = model_dir + "/0_master/" + mode + "_triplet_weights"
        set_model(path=path)


def labview_pred(img, cut_size,**kwargs):
    img = img.astype(np.float32) / 255.
    for key in kwargs:
        kwargs[key] = kwargs[key].astype(np.float32) / 255.
    heat_map = evaluate_img(img, cut_size,**kwargs)
    return heat_map

if __name__ == "__main__":
    # labview_set_model(1)
    set_model(path="model_weights/weights")

    import skimage, cv2, glob

    paths = glob.glob("test/*.jpg")

    for path in paths:
        imgs = []
        raw_img = skimage.io.imread(path, as_gray=True)
        # raw_img = cv2.GaussianBlur(raw_img,(3,3),1)
        for resize in [1,4,8]:
            img = cv2.resize(raw_img, (raw_img.shape[1] // resize, raw_img.shape[0] // resize),interpolation=cv2.INTER_LINEAR)
            img = skimage.img_as_float32(img)
            imgs.append(img)
        print(np.max(img))

        cut_size = 64
        # idx = np.random.randint(len(x_test))

        heat_map = evaluate_img(imgs[0], cut_size)#, img1=imgs[1])#, img2=imgs[2])
        show_result(imgs[0], heat_map)

