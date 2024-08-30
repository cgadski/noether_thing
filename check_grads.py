# %%
import noether
import einops
import torch as t
import matplotlib.pyplot as plt
from importlib import reload
import math
# _ = reload(noether)


# %%
model = noether.default_model()


# %%
task = noether.CategoryTask(model)


# %%
# Get activation gradients we computed.
grads = t.load('grads_1', weights_only=True) # (layer, token, n)
centered_grads = grads - grads.mean(dim=(0, 1))
input_n = grads.shape[1] // 4
input_tkns = task.make_input_tokens(input_n)


# %%
# First check: what's the scale of these gradients?
def show_gradient_scale(grads):
    sigma = t.sqrt(grads.pow(2).mean(dim=(1,2)))
    plt.plot(t.log10(sigma))
    plt.xticks(t.arange(0, 12), t.arange(1, 13).tolist())
    plt.xlabel("Layer")
    plt.ylabel("log_10(sigma)")
    plt.title("Standard deviation of gradients per layer")

show_gradient_scale(grads)

# Standard deviation has a maximum of about 1e-3 at the second layer and decreases about exponentially
# as we go deeper into the model.


# %% 
# I want to ask questions like "is this sample of vectors zero on average?" and "do these 
# two samples of vectors have the same average?"

def svd_projection(data, k=2):
    n = data.shape[0]
    centered = data - data.mean(dim=0)
    cov = centered.t() @ centered / n
    _, Q = t.lobpcg(cov, k)
    return Q

def whitening(data, k=None):
    """
    Produce 1/sqrt(cov), a linear map that "whitens" the data.

    Optionally, first perform an SVD projection into k dimensions.
    """
    n = data.shape[0]
    centered = data - data.mean(dim=0)
    cov = centered.t() @ centered / n
    if k is not None:
        S, Q = t.lobpcg(cov, k)
        return einops.einsum(1/t.sqrt(S), Q, 'k, n k -> n k')
    else: 
        Q, S, _ = t.linalg.svd(cov)
        return einops.einsum(1/t.sqrt(S), Q, 'k, n k -> n k')


def test_zero(data):
    """
    Return p value for null hypothesis that d vectors of dimension n, assumed to be
    i.i.d. with unit Gaussian variance, are drawn from a distribution of zero mean.
    """
    d, n = data.shape
    sample_mean = t.mean(data, dim=0) # under H_0, n i.i.d. numbers with std = 1/sqrt(d)
    statistic = d * sample_mean.pow(2).sum() # chi-squared with n degrees of freedom
    return 1 - t.distributions.Chi2(t.tensor(n)).cdf(statistic)

def whiten_and_test(data, k):
    projected = data @ whitening(data, k)
    return test_zero(projected)

# Sanity check. (Been a while since I did statistics.)

# This should give largeish p value, confirming the null hypothesis.
print(test_zero(t.randn((500, 10))))

# This should give a smallish p value.
print(test_zero(2 * 1/math.sqrt(500) + t.randn((500, 10))))

# Should be large.
print(whiten_and_test(0.001 * t.randn((500, 30)) @ t.randn(30, 400), 30))

# Should be small.
print(whiten_and_test(0.001 * (1 + t.randn((500, 30)) @ t.randn(30, 400)), 30))


# %%
# Are gradients centered? Apparently not. If we do an SVD projection into 10 dimensions
# and apply my questionable statistical test, we find that the norm of the average is 
# significantly larger than we would expect it to be.
[whiten_and_test(grads[l, :, :], k=10) for l in range(12)]


# %%
# Let's plot their projections into 2 dimensions.
def plot_gradients(grads, svd_per_layer=False, token_color=None, title=None, show_mean=True, layer_labels=None):
    if title is None:
        title = "Gradients by Layer"
    n_layers = grads.shape[0]
    grid_size = math.ceil(math.sqrt(n_layers))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)

    proj = whitening(einops.rearrange(grads, "layer token n -> (layer token) n"), k=2)

    for layer in range(n_layers):
        row = layer // grid_size
        col = layer % grid_size

        layer_grads = grads[layer, :, :]
        if svd_per_layer:
            proj = whitening(layer_grads, k=2)
        
        x = layer_grads @ proj[:, 0]
        y = layer_grads @ proj[:, 1]

        if show_mean:
            avg_x = x.mean()
            avg_y = y.mean()
            axs[row, col].plot(x.mean(), y.mean(), 'rx', markersize=15, markeredgewidth=3)

            magnitude = t.sqrt(avg_x**2 + avg_y**2)
            if magnitude >= 0.1:
                line_x = t.tensor([0, avg_x / magnitude * 5])
                line_y = t.tensor([0, avg_y / magnitude * 5])
                axs[row, col].plot(line_x, line_y, 'r-', linewidth=1)

        axs[row, col].scatter(x, y, s=10, c=token_color)
        if layer_labels is None:
            axs[row, col].set_title(f'Layer {layer + 1}')
        else:
            axs[row, col].set_title(f'Layer {layer_labels[layer] + 1}')

        axs[row, col].set_xlim(-5, 5)
        axs[row, col].set_ylim(-5, 5)


    for layer in range(n_layers, grid_size * grid_size):
        row = layer // grid_size
        col = layer % grid_size
        fig.delaxes(axs[row, col])

    plt.tight_layout()

# If we use the same projection for all layers, we don't get a very surprising story.
# Gradients look pretty centered and become smaller as we go through the model.
plot_gradients(grads, show_mean=False)

# However, if we choose our projection separately for each layer, we can see some
# information in the gradients.
# plot_gradients(grads, svd_per_layer=True, token_color=t.isin(input_tkns, t.tensor(task.bad_tkns)), show_mean=False)
plot_gradients(grads, svd_per_layer=True, token_color=input_tkns == task.arrow_tkn, show_mean=False)
# plot_gradients(grads, svd_per_layer=True, token_color=input_tkns == task.newline_tkn)

# Some phenomena are very easy to see and interpret. For example, on layer 12, we find that 
# the gradients for activations at arrow tokens (' ->') are tightly concentrated along a 
# single direction. That direction must be mediating whether "yes" or "no" is predicted
# as the next token.


# %%
# However, my idea isn't really to look at the clusters in gradient space at a single
# layer. Let's consider the specific hypothesis that _comparing averages of gradients_
# over subsets of our tokens might let us infer the existence of features being expressed
# at certain layers of the model.

# Let's look at the tokens in the set of "good tokens," which are lowercase letters.
# At what point in the network does allocate "features" that describe the fact that
# these tokens are lowercase, or that they should predict "yes" or "no"?

def plot_moments(feature, layers=None, show_mean=True, svd_per_layer=False):
    if layers is None:
        feature_moments = grads[:, feature, :]
    else:
        feature_moments = grads[:, feature, :][layers, :, :]
    plot_gradients(feature_moments, show_mean=show_mean, svd_per_layer=svd_per_layer)

def show_moment_correlations(feature, k=2, layers=None):
    if layers is None:
        feature_moments = grads[:, feature, :]
    else:
        feature_moments = grads[layers, feature, :]
    proj = svd_projection(einops.rearrange(feature_moments, 'layer tok n -> (layer tok) n'), k)
    projected_moments = feature_moments @ proj
    directions = projected_moments.mean(dim=1) # (layer, n)
    directions /= directions.norm(dim=1)[:, None]
    plt.matshow(einops.einsum(directions, directions, 'layer0 n, layer1 n -> layer0 layer1'))

def feature_by_tokens(tkns):
    return t.isin(input_tkns, t.tensor(tkns))

# %%
feature = feature_by_tokens(task.good_tkns)
plot_moments(feature_by_tokens(task.good_tkns))
show_moment_correlations(feature, 2)

# %%
feature = feature_by_tokens(task.good_tkns)
plot_moments(feature_by_tokens(task.good_tkns))
show_moment_correlations(feature, 40)
 
# %%
feature = feature_by_tokens([task.arrow_tkn])
plot_gradients(grads[:, feature, :][-5:, :, :], show_mean=False, svd_per_layer=False, layer_labels=[8, 9, 10, 11, 12])
# show_moment_correlations(feature, 5)


# %%
plot_gradients(grads[:, feature, :][:7, :, :], show_mean=False, svd_per_layer=False)

# %%
plot_moments(feature_by_tokens(task.good_tkns), layers=[8, 9, 10, 11])

# %%

feature_moments = grads[:, feature_by_tokens(task.good_tkns), :] # (layer, token, n)
directions = feature_moments.mean(dim=1) # (layer, n)
directions /= directions.norm(dim=1)[:, None]
plt.matshow(einops.einsum(directions, directions, 'layer0 n, layer1 n -> layer0 layer1'))
