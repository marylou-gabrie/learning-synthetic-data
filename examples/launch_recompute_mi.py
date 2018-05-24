import matplotlib.pyplot as plt
import os
from lsd.utils_io import save_recompute_mutualInfo, load_opt, load_history
from lsd.utils_plots import plot_All_multilayer
from lsd.utils_callbacks import recompute_mi, load_MI


folder = 'data/'
folder += 'expe_folder'

recompute_every = 2
recompute_mi_noise = None
recompute_inter_mi_noise = 0
up_to = 4
save = 1

opt = load_opt(folder)
mutualInfo = load_MI(folder)
mutualInfo_ = recompute_mi(opt, mutualInfo, 
                    up_to=up_to, 
                    recompute_every=recompute_every, 
                    recompute_mi_noise=recompute_mi_noise,
                    recompute_inter_mi_noise=recompute_inter_mi_noise)
history = load_history(folder)

plot_All_multilayer(history, mutualInfo_, folder)

if save == 1:
    save_recompute_mutualInfo(folder, mutualInfo_, rewrite=False,
        recompute_mi_noise=recompute_mi_noise,
        recompute_inter_mi_noise=recompute_inter_mi_noise)

    filename = folder + "/history-info_plane"
    if recompute_mi_noise is not None:
        filename += "_minoise{:0.1E}".format(recompute_mi_noise)
    if recompute_inter_mi_noise is not None:
        filename += "_internoise{:0.1E}".format(recompute_inter_mi_noise)

    filename_count = filename
    if os.path.isfile(filename + '.pdf'):
        count = 1
        while os.path.isfile(filename_count + '_{:d}.pdf'.format(count)):
            count += 1
        filename_count += "_{:d}".format(count)
        os.rename(filename + ".pdf", filename_count + ".pdf")

    plt.savefig(filename + ".pdf")
