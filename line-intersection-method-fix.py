from PIL import Image
from scipy.ndimage import median_filter
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
from itertools import zip_longest
import csv
import os
from copy import deepcopy

def get_grid(x, y, ds):
    lx = x[-1]
    ly = y[-1]
    l = np.sqrt(lx**2 + ly**2)
    cx = lx/2.0
    cy = ly/2.0
    lines = []
    c = np.cos(np.pi/4.0)
    s = np.sin(np.pi/4.0)
    R = np.array([[c, -s],
                  [s, c]])
    p = np.arange(-l, l, ds)
    #spacing by defined distance between lines
    #vertical lines
    for x_ in p:
        x1 = x_; y1 = -l
        x2 = x_; y2 = l
        lines.append([x1, y1, x2, y2])
        x1, y1, = np.dot(R, [x1 - cx, y1 - cy])
        x2, y2, = np.dot(R, [x2 - cx, y2 - cy])
        lines.append([x1+cx, y1+cy, x2+cx, y2+cy])
    #horizontal lines
    for y_ in p:
        x1 = -l; y1 = y_
        x2 = l; y2 = y_
        lines.append([x1, y1, x2, y2])
        x1, y1 = np.dot(R, [x1 - cx, y1 - cy])
        x2, y2 = np.dot(R, [x2 - cx, y2 - cy])
        lines.append([x1+cx, y1+cy, x2+cx, y2+cy])
    return lines

def get_random_lines(x, y, N):
    return

def single_sample(dat_in):
    """get a single line"""
    x = dat_in["x"]
    y = dat_in["y"]
    f = dat_in["f"]
    res = dat_in["res"]
    scale = dat_in["scale"]
    min_length = dat_in["min_length"]
    x1, y1, x2, y2 = dat_in["line"]
    #get points along the line at half pixel spacing
    ns = int(res)*int(max(abs(x2 - x1), (abs(y1 - y2))))
    xs = np.linspace(x1, x2, num=ns)
    ys = np.linspace(y1, y2, num=ns)
    #grab the ones that lie within our bounding box
    grab = ((xs >= 0) & (xs <= x[-1])) & ((ys >= 0) & (ys <= y[-1]))
    if np.all(grab == False):
        return False
    xs = xs[grab]
    ys = ys[grab]
    ns = xs.size
    xys = np.zeros((ns,2))
    xys[:,0] = ys
    xys[:,1] = xs
    zs = f(xys, 'nearest')
    #get the start and end points over which we will check
    start_id = 0
    while (zs[start_id] == zs[0]) and (start_id < ns-1):
        start_id += 1
        stop_id = ns -1
#        print(start_id, stop_id)
    while (zs[stop_id] == zs[-1]) and (stop_id > 0):
        stop_id -= 1
    if (start_id > stop_id):
        return False
    #get the sizes of pores between start and stop
#    print(start_id, stop_id)
    sample = []
    val = zs[start_id]
    count = 1
    for i in range(start_id+1, stop_id):
        if zs[i] == val:
            count += 1
        else:
            sample.append([count, val])
            val = zs[i]
            count = 1
    if not sample:
        return False
    sample = np.array(sample)
#    print(sample)
    #the two types of data
    v1 = sample[:,1].min()
    v2 = sample[:,1].max()
    grab1 = sample[:,1] == v1
    grab2 = sample[:,1] == v2
    counts_1 = sample[grab1, 0]
    counts_2 = sample[grab2, 0]
    #the size of each step of the count
    ds = np.sqrt((xs[1] - xs[0])**2 + (ys[1] - ys[0])**2)*scale
    lengths_1 = counts_1*ds
    lengths_2 = counts_2*ds
    #only keep samples that comply with minimum length
    grab1 = lengths_1 >= min_length
    lengths_1 = lengths_1[grab1]
    grab2 = lengths_2 >= min_length
    lengths_2 = lengths_2[grab2]
#    print(lengths_1, lengths_2)
    #save data
    dat = {}
    dat['xs'] = xs*scale
    dat['ys'] = ys*scale
    dat['zs'] = zs
    dat['ds'] = ds
    dat['lengths'] = [lengths_1, lengths_2]
#    print(dat['lengths'])
    return dat

def get_samples(z, N=2, ds=1, scale=1.0, min_length=1, res=4, random=False):
    """
    given image data get N random samples
    z: image data
    N: number of lines
    scale: length per pixel, for instance:
        0.52 px/um = 1.92 value in script
    min_length: minimum size of continous pixels (in units of scale)
    res: how many test points per pixel
    ncores: how many cores to use to run
    spacing: spacing between lines if we are to use regular grid (set to None for random sampling)
    """
    ny, nx = z.shape
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)
    #define interpolating function
    f = RegularGridInterpolator((y, x), z)
    data = {}
    data["global"] = {"x":scale*x, "y":scale*y, "z":z}
    data["samples"] = []
    if random:
        lines = None
    else:
        lines = get_grid(x, y, ds/scale)
    dat_in = []
    for i, line in enumerate(lines):
        dat = {"x":x, "y":y, "f":f, "scale":scale, "min_length":min_length, "res":res, "line":line}
        dat_in.append(dat)
#    print(dat_in)
    for i, d in enumerate(dat_in):
        print("line %i"%(i+1))
        dat = single_sample(d)
        if dat:
            data["samples"].append(dat)
#            print(data)
#        else:
#            print("not dat")
    # print("dat:", dat)
#    print("dat_in:", dat_in)
    # print(data)
    del dat_in
    return data

def collate(data, bin_width):
    """collate the data for saving"""
    # print(data)
    data["collated"] = {}
    dc = data["collated"]
    names = ['lo', 'hi']
    for i, name in enumerate(names):
        lengths = []
        for d in data["samples"]:
            lengths.extend(d["lengths"][i])
#            print(lengths)
        analysis(lengths, bin_width, dc, name) 
#    dc["relative density"] = [data["relative density"]]
    return

def analysis(lengths, bin_width, dc, name):
    """statistics"""
    if not lengths:
        raise RuntimeError("lengtht list is empty")
        #raw data
    bins = np.arange(0, np.max(lengths), bin_width)
    weights, bins = np.histogram(lengths, bins)
    dc["lengths_"+name] = lengths
    dc["weights_"+name] = weights.tolist()
    dc["bins_"+name] = bins.tolist()
    #relweights
    relweights = weights.copy()
    relweights = relweights/float(np.sum(weights))
    dc["relweights_"+name] = relweights.tolist()
    #approximation of contribution to the total
    #length of each bin
    mean_bins = (bins[1::] + bins[0:-1])/2.0
    contribution = weights*mean_bins
    contribution /= np.sum(contribution)
    dc["contribution_"+name] = contribution.tolist()
    wm = np.sum(mean_bins*contribution)
    dc["c_mean_"+name] = [wm]
    dc["c_std_"+name] = [np.sqrt(np.sum(contribution*(mean_bins- wm)**2))]
    return 

def save_csv(data, name):
    """save to csv format"""
    to_write = []
    items = data["collated"].keys()
    items = [it[::-1] for it in items]
    items = sorted(items)
    items = [it[::-1] for it in items]
    for d in items:
        to_write.append([d] + data["collated"][d])
    to_write = list(zip_longest(*to_write, fillvalue=''))
    f = open(name, 'w')
    writer = csv.writer(f)
    writer.writerows(to_write)
    f.close()
    return

def plot(q):
    """data visualisation"""
    data = q["data"]
    fig = plt.figure(figsize=(8,12))
    gs = gridspec.GridSpec(5,2)
    ax1 = fig.add_subplot(gs[0:3,:])
    x = data["global"]["x"]
    y = data["global"]["y"]
    z = data["global"]["z"]
    cb = ax1.pcolormesh(x,y,z, cmap="bone", vmin=z.min(), vmax=z.max()) #image plot
    plt.colorbar(cb, ax=ax1)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(y[0], y[-1])
    ax1.set_aspect(1)
    for d in data["samples"]:
        ax1.plot(d['xs'], d['ys'], 'r-', lw = q["line_width"], alpha = q["line_opacity"]) #line plot
    dc = data["collated"]
    
    print("dc keys =",dc.keys())
    
    for i,n in enumerate(["lo", "hi"]):
        bins = dc["bins_"+n]
        contribution = dc["contribution_"+n]
        weights = dc["weights_"+n]
        ax = fig.add_subplot(gs[-2, i])
        ax.bar(bins[0:-1], weights, bins[1] - bins[0])
        ax.set_title("raw "+n)
        ax = fig.add_subplot(gs[-1, i])
        ax.bar(bins[0:-1], contribution, bins[1]-bins[0])
        ax.set_title("contribution % "+n)
    fig.tight_layout()
    return fig

def listdir_fullpath(d):
    return [os.path.join(d,f) for f in os.listdir(d)]

def process(q):
    name = q["file"]
    print("file",name)
    im = Image.open(name).convert('L')
    z = np.array(im)
    im.close()
    if q["image_subset"]:
        ni, nj = q["image_subset"]
        z = z[0:ni, 0:nj]
    #histogram bin widths that will be saved
    bin_width = q["bin_width"]
    #remove speckles
    print("filtering image...", end = '')
    min_size = q["min_size"]
    z = median_filter(z, size = min_size)
    print("done")
    #change into binary format
    print("thresholding...", end = '')
    threshold = q["threshold"]
    z = np.where(z>threshold, 1,0)
    print("done")
    print("analysing...", end = '')
    data = get_samples(z, ds=q["ds"], scale=q["scale"], min_length=q["min_length"], random=False)
    # print(data)
    print("done")
    #save data
    q["data"] = data
    print("collating...", end = '')
    collate(data, bin_width)
    print("done")
    #change where to save
    folder, name = os.path.split(name)
    name = os.path.join(q["save_folder"], name)
    print("saving...", end = '')
    csv_name = name.replace(".tif", ".csv")
    save_csv(data, csv_name)
    print("done")
    print(csv_name)
    print("saving figure...", end = '')
    fig = plot(q)
    fig_name = name.replace(".tif", ".png")
    fig.savefig(fig_name, dpi=q["dpi"])
    print("done")
    print(fig_name, " processing complete")
    del z
    return 

def preprocess(Q):
    QQ = []
    for q in Q:
        #folder generation
        q["save_folder"] = os.path.join(q["folder"], q["save_folder"])
        if not os.path.exists(q["save_folder"]):
            os.makedirs(q["save_folder"])
        #get a list of all files in the folder
        files = listdir_fullpath(q["folder"])
        for f in files:
            for tp in [".png", ".tif"]:
                if tp in f:
                    qq = deepcopy(q)
                    qq["file"] = f
                    for c in q["custom"]:
                        if c["image"] in f:
                            for key in c.keys():
                                if key != "image":
                                    qq[key] = c[key]
                    del qq["custom"]
                    QQ.append(qq)
                    break
    return QQ

def run(Q):
    """run this instruction set"""
    for q in Q:
        process(q)
    return

def group_statistics(Q, bin_width):
    """calculate statistics for each group"""
    groups = {}
    for q in Q:
        folder = q["folder"]
        if folder in groups:
            groups[folder].append(q)
        else:
            groups[folder] = [q]
    data = {}
    data["collated"] = {}
    dc = data["collated"]
    names = ["lo", "hi"]
    for folder in groups.keys():
        group = groups[folder]
        n_b = 0
        n_w = 0
        for i,name in enumerate(names):
            lengths = []
            for g in group:
                n_b += g["data"]["number_black"]
                n_w += g["data"]["number_white"]
                for d in g["data"]["samples"]:
                    lengths.extend(d["lengths"][i])
            analysis(lengths, bin_width, dc, name)
        dc["relative density"] = [n_b/float(n_b+n_w)]
        save_folder = group[0]["save_folder"]
        name = os.path.join(save_folder, "all_data.csv")
        save_csv(data, name)


if __name__ == "__main__":
    bin_width = 10#20
    df = {}
    df["bin_width"] = bin_width
    df["min_size"] = 5 #15
    df["threshold"] = 150
    df["ds"] = 38
    df["scale"] =  1.0 #2.1367521
    df["min_length"] = 2.0 #5.0
    df["dpi"] = 600
    df["custom"] = []
    df["line_width"] = 0.3
    df["line_opacity"] = 0.5
    df["image_subset"] = [] #for full image set this to [], otherwise [200, 200]

    Q = []
    q = deepcopy(df)
    q["folder"] = r"."
    q["save_folder"] = "data"
    # q["threshold"] = 160 #code for altering the threshold for this folder
    q["custom"] = [{"image":"0-lx.tif", "threshold":150}]
    Q.append(q)
    Q = preprocess(Q)
    run(Q)
    group_statistics(Q, bin_width)
    print("DONE")





