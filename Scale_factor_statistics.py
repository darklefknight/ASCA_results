import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt


if __name__ == "__main__":
    PATH = "/home/mpim/m300517/MPI/finished/ASCA_results/"
    NAME = "CloudCoverage_20170601_sf"
    nc100 = Dataset(PATH+NAME+"100.nc")
    nc50 = Dataset(PATH + NAME + "50.nc")
    nc25 = Dataset(PATH + NAME + "25.nc")
    nc10 = Dataset(PATH + NAME + "10.nc")

    ncs = [nc100,nc50,nc25,nc10]
    names = ["nc100","nc50","nc25","nc10"]

    fig1 = plt.figure(figsize=(16,9))
    ax = fig1.add_subplot(2,1,1)
    ax2 = fig1.add_subplot(2,1,2)

    for nc, name in zip(ncs,names):
        cc = nc.variables["cc"][:]
        ax.plot(cc, label=name)
        if nc != nc100:
            ax2.plot(nc100.variables['cc'][:]-cc, label=name)

    ax.grid()
    ax.legend(loc="best")
    ax.set_title("Cloudcoverage")
    ax.set_ylabel("cc in percent")

    ax2.set_title("Cloudcoverage diviation from Full resolution.")
    ax2.set_ylabel("Diviation from full resolution in percent-points")
    ax2.set_xlabel("Timestep")
    ax2.grid()
    ax2.legend(loc="best")
    plt.savefig("ScaleFactorStatistics.png")
    fig1.show()





    for nc in ncs:
        nc.close()

