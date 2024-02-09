# imports here 

def render_ratio_corr_for_cells(
    x, 
    y,
    d0, 
    dS, 
    num_rois=30000, 
    nnpop=10, 
    rnpop=10, 
    seed=None, 
    tag=None, 
    sdir=None, 
    figsize=(20, 20), 
    edgecolor='white', 
    marker_size=10, 
    radial_scale=10,
    circle_patch=False,
    roc = roc0,
):
    fig0 = plt.figure(figsize=figsize)
    custom_norm = TwoSlopeNorm(vcenter=1, vmin=roc.min(), vmax=roc.max())
    ax = plt.axes()
        
    ps0=plt.scatter(
            x[:num_rois],
            y[:num_rois],
            marker=".",
            norm=custom_norm,
            cmap="rainbow",
            s=marker_size,
            c=roc,
    )

    if circle_patch:
        plt.gca().add_patch(plt.Circle((250, 50), 
                                       radius=radial_scale, 
                                       facecolor='none', 
                                       edgecolor=edgecolor, 
                                       linewidth=2.5
                                    )
                    )

    plt.colorbar(ps0, spacing='proportional', shrink=0.5).ax.set_yscale('linear')
    plt.xlabel("ROI X Positions", fontsize=20)
    plt.ylabel("ROI Y Positions", fontsize=20)
    #plt.grid(color='white')
    plt.margins(x=0, y=0)
    plt.title(
        f"{tag}:correlation ratios for {num_rois} cells with NN ROIs:{nnpop} to RN ROIs:{rnpop} seed:{seed}",
        fontsize=20,
    )
    ax.set_facecolor("black")
    plt.tight_layout()
    if sdir:
        plt.savefig(
            f"{sdir}Ratiocorrs_{tag}_ROIs:{num_rois}_NN:{nnpop}_seed:{seed}_RN:{rnpop}.png",
        )
    else:
        plt.grid(color='white')
        plt.show()
        
    plt.close()

def render_signal_correlations_for_roi(d0, roi, nn_arr, rn_arr, roi_roc, nn_corr, rn_corr):
    """ Plots the signals or near and random neighs for visual comparisions"""
    # sort rois for plot layout
    ls_roi_idx  = sort_rois_in_plot_layout(roi, nn_arr, rn_arr)

    # compute nrows for plot
    nrows = int(len(ls_roi_idx)/2)
    ls0_iter = iter(ls_roi_idx)
    
    # sort correlations
    ls_roi_corr = sort_rois_in_plot_layout(roi_roc, nn_corr, rn_corr)
    ls_roi_corr_iter = iter(ls_roi_corr)

    fig, axs = plt.subplots(ncols=2, 
                           nrows=nrows, 
                           figsize=(15, 4),
                           layout="constrained",
                )
    # for each Axes, add an artist, in this case a nice label in the middle...
    for row in range(4):
        for col in range(2):
            idx = next(ls0_iter) # get index of next plot
            cori = round(next(ls_roi_corr_iter), 3)

            if (row > 0) & (col == 1):
                # plotting random neighbours
                axs[row, col].plot(d0[:, idx][80:1880], linewidth=0.5, label=f'{idx}:{cori}',  color='orange' )

            elif row == 0:
                # plotting main signal
                axs[row, col].plot(d0[:, idx][80:1880],  linewidth=0.5, color= 'blue', label=f'{idx}:{cori}')

            else:
                # plotting near neighbours
                axs[row, col].plot(d0[:, idx][80:1880],  linewidth=0.5, color='red', label=f'{idx}:{cori}')
            axs[row, col].set_facecolor('#eafff5')
            axs[row, col].text(-0.5, -0.15, s=f'{idx}:{cori}', color='red')
            axs[row, col].legend(handlelength=0, fontsize='large', loc='lower left', labelcolor='white', frameon=False)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].margins(x=0)
            axs[row, col].annotate(f'', (0.5, 0.5),
                                transform=axs[row, col].transAxes,
                                ha='center', va='center', fontsize=18,
                                color='darkgrey')
                

    axs[3, 0].set_xticks([0, 1800])
    axs[3, 1].set_xticks([0, 1800])
    axs[0, 0].set_title('Main ROI, ROI IDX : ROC', loc='left', fontsize='medium') 

    #axs[1, 0].set_title('Nearest Neighbours', loc='left', fontsize='medium') 
    #axs[1, 1].set_title('Random Neighbours', loc='right', fontsize='medium') 
    fig.suptitle("Correlations Sanity Check: Rendering correlations for given ROI")
    plt.show()

