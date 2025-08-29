import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

SOFTMAX_KQV = 'softmax_layer_head_kqv.pkl'
GLA_KQV = 'gla_layer_head_kqv_pruned.pkl'
CMAP = plt.get_cmap('coolwarm', 7)
COLORS = [CMAP(2), CMAP(5)]
BACKGROUND_COLOR = '#FCFBF8'

def process_layer_head_kqv(layer_head_kqv):
    layer_head_kqv_flattened = {}
    for layer_head, kqv in layer_head_kqv.items():
        k = kqv['k']
        q = kqv['q']
        v = kqv['v']
        head_dim = k.shape[-1]
        layer_head_kqv_flattened[layer_head] = {
            'k': k.reshape(-1, head_dim),
            'v': v.reshape(-1, head_dim),
            'q': q.reshape(-1, head_dim)
        }
    return layer_head_kqv_flattened

def do_tsne(k: torch.Tensor, q: torch.Tensor, model_type: str, layer_idx: int, head_idx: int):
    """
    Plot k and q vectors in TSNE space. Need to maybe play around with perplexity
    """
    num_samples = k.shape[0]
    k = k.cpu().numpy()
    q = q.cpu().numpy()
    combined_data = np.vstack([k, q])
    labels = np.array(['k'] * num_samples + ['q'] * num_samples)

    # Apply tsne
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined_data)

    # Plot
    fix, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(['k', 'q']):
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            color=COLORS[i],
            label=f'{label} vectors',
            alpha=0.6,
            s=10
        )
    ax.legend()
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_facecolor(BACKGROUND_COLOR)
    plt.title(f'{model_type} t-SNE of K and Q vectors')
    plt.savefig(f'{model_type}_tsne_kq_({layer_idx}, {head_idx}).png', dpi=300)

def do_pca(k: torch.Tensor, q: torch.Tensor, model_type: str, layer_idx: int, head_idx: int, n_components: int=2):
    """
    Plot k and q vectors in TSNE space. Need to maybe play around with perplexity
    Get singule values and plot them as well
    """
    num_samples = k.shape[0]
    k = k.cpu().numpy().astype(np.float32)
    q = q.cpu().numpy().astype(np.float32)
    combined_data = np.vstack([k, q])
    labels = np.array(['k'] * num_samples + ['q'] * num_samples)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(combined_data)

    # Get singular values
    U_q, s_q, Vt_q = np.linalg.svd(q, full_matrices=False)
    U_k, s_k, Vt_k = np.linalg.svd(k, full_matrices=False)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: PCA scatter plot
    for i, label in enumerate(['k', 'q']):
        mask = labels == label
        ax1.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            color=COLORS[i],
            label=f'{label} vectors',
            alpha=0.6,
            s=10
        )

    ax1.legend()
    ax1.set_title(f'{model_type} PCA visualization of Q and K vectors')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_facecolor(BACKGROUND_COLOR)

    # Right plot: Singular values
    ax2.plot(s_q, 'r-', label='Q matrix singular values', linewidth=2)
    ax2.plot(s_k, 'b-', label='K matrix singular values', linewidth=2)
    ax2.set_title('Singular Values')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Singular Value')
    ax2.legend()
    ax2.set_facecolor(BACKGROUND_COLOR)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_type}_pca_kq_({layer_idx}, {head_idx}).png', dpi=300)

    # Print variance info
    print(f'({model_type}) PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance')
    print(f'({model_type}) PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance')
    print(f'({model_type}) Total variance explained: {pca.explained_variance_ratio_.sum():.2%}')


if __name__ == '__main__':
    LAYER_HEADS = [(0, 0)]
    for model_type, data_path in [
        ('gla', GLA_KQV),
        ('softmax', SOFTMAX_KQV)
    ]:
        with open(data_path, 'rb') as f:
            layer_head_kqv = pickle.load(f)

        # Combine the q and k and v (merge along the batch and seq_len dimension)
        layer_head_kqv_combined = process_layer_head_kqv(layer_head_kqv)

        for layer, head in LAYER_HEADS:
            try:
                kqv = layer_head_kqv_combined[(layer, head)]
                do_tsne(kqv['k'], kqv['q'], model_type, layer, head)
                do_pca(kqv['k'], kqv['q'], model_type, layer, head)
            except KeyError:
                print(f'Layer head ({layer}, {head}) data not found')
                continue