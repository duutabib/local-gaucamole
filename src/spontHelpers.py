import statistics as sts
from math import ceil

import fastcluster as fc
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from pylab import *
from scipy.cluster import hierarchy as H
from scipy.sparse.linalg import eigsh
from scipy.stats import kurtosis, skew


# low-d tuning curves
def low_dim_tuning_curves(xx, nn):
    _n = 2**12
    y = np.cos(xx + 2 * np.pi * (nn / _n))

    return y


# cosine tuning curves
def cool_tuning_curves(xx, alpha, nn):
    y = np.cos(nn * xx)
    y /= (nn + 1) ** (alpha / 2)

    return y


# sine tuning curves
def soul_tuning_curves(xx, alpha, nn):
    y = np.sin(nn * xx)
    y /= (nn + 1) ** (alpha / 2)

    return y


# Efficient coding tuning curves
def cool_eff_tuning_curves(xx, nn):
    N = xx.__len__()
    if nn >= N:
        nn = nn % N
    results = np.zeros((N,))
    results[nn] = 1

    return results


def random_projections(x):
    # gen random basis based on the shape[1]
    y = normalize_activity(x)
    n = y.shape[1]
    _basis = np.random.randn(3, n)
    results = _basis @ y
    return results


def render_projections(pop, alpha=0.7, projection="3d"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    ax.plot(pop[0], pop[1], pop[2], color="black")
    ax.plot(pop[0], pop[1], pop[2].min(), color="grey", alpha=alpha)

    return None


def compute_vars(x_tc):
    #  compute svd of activity
    y_tc = normalize_activity(x_tc)
    s, _ = eigsh(y_tc, k=4000)  # compute eigenvalues
    # square, and normalize
    s = s.sqrt()
    ss = s / s.sum()

    return ss


def test_vars(x):
    # compute the singular values of activity
    y = normalize_activity(x)
    _, s, _ = svd(x)
    # square for eigenvalues
    s = s**2
    ss = s / s.sum()

    return ss


def render_variances(ss):
    plt.loglog(ss)
    plt.xlim(0, 10**4)
    plt.ylim(10**-4, 10**0 + 20e-2)
    plt.xlabel("Dimensions")
    plt.ylabel("Variances")
    plt.grid(True)

    return None


def normalize_activity(x):
    y = x - x.mean()

    return y


def get_alpha_fit(ss, trange):
    """fit alpha to variance curve
    and predict values based on y
    """

    log_vars = np.log(np.abs(ss))  # log of vars
    y = log_vars[trange][:, np.newaxis]  # extend dim by 1 col
    nt = trange.size  # get size of subset for computation
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    b = np.linalg.solve(x.T @ x, x.T @ y).flatten()  # least square solution
    alpha = b[0]

    complete_range = np.arange(0, ss.size).astype(int) + 1
    xx = np.concatenate(
        (-np.log(complete_range)[:, np.newaxis], np.ones((ss.size, 1))), axis=1
    )
    y_pred = np.exp((xx * b).sum(axis=1))

    return alpha, y_pred


""""Stringer tools"""


def ssplit(x):
    """Returns a new array of split
    spontaneous activity.
    (2, n, p)
    """
    nr = x.shape[0]

    nro = x.shape[0]
    nco = x.shape[1]

    newarr = np.empty((2, nro, nco))

    snr = math.ceil(nr / 2)

    p0 = x  # [:snr, :]
    p1 = x  # [snr:nr, :]

    nrp0 = p0.shape[0]
    ncp0 = p0.shape[1]

    nrp1 = p1.shape[0]
    ncp1 = p1.shape[1]

    newarr[0, :, :] = p0
    newarr[1, :, :] = p1

    return newarr


def unbox(x):
    """returns 3 outputs resp (response), spon (activity),
    istim (index of stimuli)
    x : data file
    """
    resp = x["stim"]["resp"]
    spon = x["stim"]["spont"]
    istm = (x["stim"]["istim"]).astype(np.int32)
    istm -= 1
    nimg = istm.max()
    resp = resp[istm < nimg, :]
    istm = istm[istm < nimg]

    return resp, spon, istm


def denoise_resp(r, sp):
    """returns a mean shifted response expressed
    in the spont activity basis

    r: response
    sp: spontaneous activity
    """
    _u = sp.mean(axis=0)
    _sd = sp.std(axis=0) + 1e-6
    r = (r - _u) / _sd
    sp = (sp - _u) / _sd
    sv, u_ = eigsh(sp.T @ sp, k=32)
    r = r - (r @ u_) @ u_.T
    r -= r.mean(axis=0)

    return r


def dupSignal(x, idx):
    """returns array of multiple responses of repeated stimulus
    nimg: # of  natural images
    nn: number of neurons
    idx: stim indexes
    x: response
    """
    nimg = 2800  # num of images
    nn = x.shape[1]
    y = np.zeros((2, nimg, nn), np.float64)  # def internal
    inan = np.zeros((nimg,), bool)  # def internal
    for n in range(nimg):
        ist = (idx == n).nonzero()[0]
        il = ist[: int(ist.size / 2)]
        i2 = ist[int(ist.size / 2) :]
        # check if two repeats of stim
        if np.logical_or(i2.size < 1, il.size < 1):
            inan[n] = 1
        else:
            y[0, n, :] = x[il, :].mean(axis=0)
            y[1, n, :] = x[i2, :].mean(axis=0)

        # rm single responses
    y = y[:, ~inan, :]

    return y


def pad_vector(v, desired_length, pad_with=np.nan):
    v_new = np.zeros((desired_length,))
    v_new[: len(v)] = v
    v_new[len(v) :] = pad_with

    return v_new


def abib_cdf(r, r_min=-np.inf, **kwargs):
    r_kept = r[r > r_min]
    print(f"Dropped {len(r) - len(r_kept)} values less than {r_min=}")
    x = sorted(r_kept)
    y = arange(1, len(r_kept) + 1) / (len(r_kept))
    plot(x, y, **kwargs)
    xlabel("Samples")
    ylabel("Cumulative Probability")

    return r_kept


def abib_metric(X):
    n = X.shape[0]
    X2 = X**2
    EX2 = mean(sum(X2, axis=0))
    r = EX2 / n

    return r


def orlicz(x, psi, ts=logspace(-1, 1, 10000)):
    for t in ts:
        d = psi(abs(x) / t)
        m = mean(d)
        if m <= 1:
            return t
    return np.inf


def sub_g(x, **qargs):
    f = lambda x: exp(x**2) - 1

    return orlicz(x, f, **qargs)


def sub_e(x, **qargs):
    f = lambda x: exp(x) - 1

    return orlicz(x, f, **qargs)


def sub_g_(xs, t=linspace(0.01, 1000, 10000)):
    sxs = xs**2
    for j in t:
        d = exp(sxs / j**2)
        if mean(d) <= 2:
            return j

    return np.inf


def sub_e_(xs, t=linspace(0.01, 1000, 100000)):
    for j in t:
        d = exp(abs(xs) / j)
        m = mean(d)
        if m <= 2:
            return j

    return np.inf


def ro_orders(X, *args):
    "Returns a structured dataset via Hierarchical clustering (scipy)"
    Zrow = fc.linkage(clip(X, 0, inf), method="average", metric="euclidean")
    row_order = H.leaves_list(Zrow)
    Zcol = fc.linkage(clip(X, 0, inf).T, method="average", metric="euclidean")
    col_order = H.leaves_list(Zcol)

    return row_order, col_order


def imboard(resp, row_order, col_order, ss_resp_, ypred, alpha_resp, c, filename):
    # data ppp
    daata = sqrt(clip(resp[row_order][:, col_order][row_order, :], 0, inf))

    alf = round(alpha_resp, 2)
    c = -1 * round(c, 2)

    print("intercept etc... ")

    # covariances
    R = clip(resp, 1, inf)
    C = corrcoef(R)
    Ct = corrcoef(R.T)

    print("computing covariance etc...")

    # xy values and vertical lines
    xvalues = mean(resp, axis=0)
    yvalues = arange(resp.shape[1])
    muinf = percentile(xvalues, 90)
    low_muinf = percentile(xvalues, 1)
    print("vertile ptiles, also for hist")
    print("90 tile", muinf)
    print("1 tile", low_muinf)
    mu0 = mean(xvalues)
    muline = mu0
    sdline0 = mu0 + std(xvalues)
    sdline1 = mu0 - std(xvalues)

    # mean, std+, sdt-, line per stimulus
    mu = mean(resp, axis=1)
    msinf = percentile(mu, 90)
    low_msinf = percentile(mu, 1)
    print("stimlus ptiles")
    print("90 tile", msinf)
    print("1 tile", low_msinf)
    nmean = len(mu)
    mumean = np.mean(mu)
    mline = [mumean] * nmean
    sline0 = [mumean + np.std(mu)] * nmean
    sline1 = [mumean - np.std(mu)] * nmean

    # logr
    logr = log10(clip(resp, 1, inf))

    # plotting
    fig0 = plt.figure(constrained_layout=True, figsize=(10, 10))
    gs = GridSpec(8, 8, figure=fig0)

    # data
    ax0_data = fig0.add_subplot(gs[0:2, :4])
    matshow(daata.T, cmap=cm.turbo, fignum=False, aspect="auto")
    title("Activity of ~10K Neurons")
    xlabel("Image")
    ylabel("Neuron")

    # mean activity of neurons
    ax0_mnr = fig0.add_subplot(gs[0:2, 4:6], sharey=ax0_data)
    plot(xvalues, yvalues, color="darkblue")
    axvline(muline, label="mean", color="red", linestyle="dotted")
    axvline(sdline0, label="+std", color="blue", linestyle="--")
    axvline(sdline1, label="-std", color="orange", linestyle="--")
    xlim(-0.2, 300)
    title("Mean Activity of Neurons")
    xlabel("Mean activity")
    ylabel("Neurons")

    # mean distribution across stimulus
    ax0_mir = fig0.add_subplot(gs[2:4, :4])
    plot(np.sort(mean(resp, axis=1), axis=None), color="darkblue")
    plot(mline, label="mean", color="red", linestyle="dotted")
    plot(sline0, label="+std", color="blue", linestyle="--")
    plot(sline1, label="-std", color="orange", linestyle="--")
    xlabel("Image")
    ylabel("Activity")
    ylim(low_msinf, msinf)
    legend()
    title("Mean Activity per Stimulus ")

    # Histogram of the mean response across neurons
    ax0_mir = fig0.add_subplot(gs[2:4, 4:6])
    hist(mean(resp, axis=0), bins=100, color="darkblue")
    xlabel("Mean per neuron")
    ylabel("Count")
    xlim(0, 300)
    ylim(0, 2500)
    title("Mean Distribution of Neurons")

    # Corr Neuron vs Neuron
    ax0_mir = fig0.add_subplot(gs[4:6, 0:2])
    matshow(
        clip(C[row_order][:, row_order], 0, inf),
        cmap=cm.turbo,
        fignum=False,
        aspect="auto",
    )
    xticks(rotation=30)
    xlabel("Image")
    ylabel("Image")
    title("Covariance")

    # Corr image of the mean response across neurons
    ax0_mir = fig0.add_subplot(gs[4:6, 2:4])
    matshow(
        clip(Ct[col_order][:, col_order], 0, inf),
        cmap=cm.turbo,
        fignum=False,
        aspect="auto",
    )
    xticks(rotation=30)
    xlabel("Neuron")
    ylabel("Neuron")
    title("Covariance")

    # Power law of response
    ax0_ivl = fig0.add_subplot(gs[4:6, 4:6])
    loglog(
        np.arange(0, ss_resp_.size) + 1,
        ss_resp_ / ss_resp_.sum(),
        label="observed",
        color="darkblue",
    )
    loglog(
        np.arange(0, ss_resp_.size) + 1,
        ypred,
        label=f"fit =-({alf}x + {c})",
        color="orange",
    )
    xlabel("Dimensions")
    ylabel("Neuron")
    text(x=10**1, y=0.1, s=f"alpha={alf}")
    title("Response Power Law")
    legend(loc="lower left", fontsize="small")

    # Skewness (third order moment)  mean vs std
    ax0_ivl = fig0.add_subplot(gs[6:8, 0:2])
    plot(mean(logr, axis=0), std(logr, axis=0), ".", color="darkblue")
    xlabel("Mean")
    ylabel("Std")
    xlim(0.00, 2.00)
    ylim(0.00, 2.00)
    title("Std vs Mean")

    # mean vs Skewness
    ax0_ivl = fig0.add_subplot(gs[6:8, 2:4])
    plot(mean(logr, axis=0), skew(logr), ".", color="darkblue")
    xlabel("Mean")
    ylabel("Skewness")
    xlim(0.00, 2.00)
    ylim(-1.50, 5.00)
    title("Skewness vs Mean ")

    # Skewness vs std
    ax0_ivl = fig0.add_subplot(gs[6:8, 4:6])
    plot(std(logr, axis=0), skew(logr), ".", color="darkblue")
    xlabel("Std")
    ylabel("Skewness")
    xlim(0.00, 2.00)  # std
    ylim(-1.50, 5.00)  # Skewness
    title("Skewness vs Std")

    fig0.suptitle(filename)

    return None
