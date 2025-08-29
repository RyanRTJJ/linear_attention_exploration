import os
import pickle
from typing import List, Optional, Tuple

import datasets
import fla
from fla.models import GLAConfig
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

SOFTMAX_ATTENTION_MODEL_NAME = 'fla-hub/transformer-1.3B-100B'
LINEAR_ATTENTION_MODEL_NAME = 'fla-hub/gla-1.3B-100B'
DEFAULT_TORCH_DTYPE = torch.float16

def get_dataset():
    repo_name = 'THUDM/LongBench'
    dataset_name = 'qasper'
    data = datasets.load_dataset(repo_name, dataset_name, split='test')
    return data

def sanity_check_model(model):
    """
    Make sure things like model datatype are expected
    """
    assert model.dtype == DEFAULT_TORCH_DTYPE, f'Non-{DEFAULT_TORCH_DTYPE} found: {model.dtype}'

def get_linear_attn_model_tokenizer(model_name: str = LINEAR_ATTENTION_MODEL_NAME):
    """
    Gets gated linear attention 1.3B params by defaultl only works if you have fla

    GLAForCausalLM(
    (model): GLAModel(
        (embeddings): Embedding(32000, 2048)
        (layers): ModuleList(
        (0-23): 24 x GLABlock(
            (attn_norm): RMSNorm(2048, eps=1e-06)
            (attn): GatedLinearAttention(
            (q_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (g_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (gk_proj): Sequential(
                (0): Linear(in_features=2048, out_features=16, bias=False)
                (1): Linear(in_features=16, out_features=1024, bias=True)
            )
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (g_norm_swish_gate): FusedRMSNormGated(512, eps=1e-06, activation=swish)
            )
            (mlp_norm): RMSNorm(2048, eps=1e-06)
            (mlp): GatedMLP(
            (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
            (swiglu_linear): SwiGLULinear()
            )
        )
        )
        (norm): RMSNorm(2048, eps=1e-06)
    )
    (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
    )
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DEFAULT_TORCH_DTYPE
    ).cuda()
    return model, tokenizer

def get_softmax_attn_model_tokenizer(
        model_name: str = SOFTMAX_ATTENTION_MODEL_NAME
    ):
    """
    Gets vanilla softmax attention model 1.3B params by default. Looks like this:


    TransformerForCausalLM(
    (model): TransformerModel(
        (embeddings): Embedding(32000, 2048)
        (layers): ModuleList(
        (0-23): 24 x TransformerBlock(
            (attn_norm): RMSNorm(2048, eps=1e-06)
            (attn): Attention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary): RotaryEmbedding(dim=64, base=10000.0, interleaved=False, pos_idx_in_fp32=True)
            )
            (mlp_norm): RMSNorm(2048, eps=1e-06)
            (mlp): GatedMLP(
            (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
            (swiglu_linear): SwiGLULinear()
            )
        )
        )
        (norm): RMSNorm(2048, eps=1e-06)
    )
    (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
    )
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DEFAULT_TORCH_DTYPE
    ).cuda()
    return model, tokenizer

def get_attn_state_vectors(
        state,
        model,
        tokenizer,
        dataset: list,
        layer_head_pairs: int | List[Tuple[int]],
        sequence_idxs: int | List[int],
        max_tokens: Optional[int] = None,
    ) -> dict:
    """
    Gets the key / query vectors of the model, when run on the dataset, for sequence_idx
    in `sequence_idxs`, at layer_head_pairs

    Returns a map: {
        sequence_idx: {
            (layer, head): torch.Tensor of shape (seq_len, dim)
        }
    }
    """
    dataset_size = len(dataset)
    config = model.config

    num_layers = config.num_hidden_layers
    num_heads = config.num_heads

    # Sanity check
    assert state in ['keys', 'values', 'queries']

    # Get sequence idxs
    if isinstance(sequence_idxs, int):
        assert sequence_idxs <= dataset_size, f'sequence_idxs cannot be > dataset_size got {sequence_idxs}'
        sequence_idxs = random.sample(range(dataset_size), min(sequence_idxs, dataset_size))
        random_subset = [dataset[i] for i in sequence_idxs]
        random_subset_contexts = [data['context'] for data in random_subset]
    else:
        assert len(sequence_idxs) <= dataset_size
        random_subset = [dataset[i] for i in sequence_idxs]
        random_subset_contexts = [data['context'] for data in random_subset]

    # Get layer_heads_pairs
    if isinstance(layer_head_pairs, int):
        # Randomly sample some layer_head_pairs
        assert layer_head_pairs > 0
        all_possible_pairs = [(i, j) for i in range(num_layers) for j in range(num_heads)]
        layer_head_pairs = random.sample(all_possible_pairs, layer_head_pairs)
    assert len(layer_head_pairs) > 0, 'Provide non-empty layer_head_pairs to visualize attention for'
    for layer_idx, head_idx in layer_head_pairs:
        assert layer_idx < num_layers and head_idx < num_heads, f'Invalid layer_head_pair: {layer_head_pair}'

    truncation = True if max_tokens else False

    inputs = tokenizer(
        random_subset_contexts,
        return_tensors='pt',
        truncation=truncation,
        max_length=max_tokens,
    ).to(model.device)

    layer_kqv = {}
    # key: (layer_idx, head_idx) -> value: {
    #   q: tensor of shape (batch_size, num_heads, head_dim),
    #   k: ...
    #   v: ...
    # }
    
    def create_proj_hook(proj_type: str, layer_idx: int):
        def hook_fn(module, _inputs, _output):
            if layer_idx not in layer_kqv:
                layer_kqv[layer_idx] = {}

            print(f'\n{proj_type}:')
            print(_output)
            # Store projection output.
            # The shapes for outputs are (batch, seq_len, num_heads * head_dim)
            # Goal is to get them all to (batch, num_heads, seq_len, head_dim)
            # print(f'proj_type: {proj_type}, output_shape: {_output.shape}')
            batch_size, seq_len, _ = _output.shape
            head_dim = _output.shape[-1] // num_heads
            layer_kqv[layer_idx][proj_type] = _output.detach().reshape(
                batch_size, seq_len, num_heads, head_dim
            ).transpose(1, 2)
        
        return hook_fn

    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        # Hook q_proj
        q_hook = layer.attn.q_proj.register_forward_hook(
            create_proj_hook('q', i)
        )
        hooks.append(q_hook)
        
        # Hook k_proj  
        k_hook = layer.attn.k_proj.register_forward_hook(
            create_proj_hook('k', i)
        )
        hooks.append(k_hook)
        
        # Hook v_proj
        v_hook = layer.attn.v_proj.register_forward_hook(
            create_proj_hook('v', i)
        )
        hooks.append(v_hook)

        
    # Do forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    layer_head_kqv = {}
    for l in range(num_layers):
        layer_head_kqv[l] = {}
        this_layer_kqv = layer_kqv[l]
        for h in range(num_heads):
            layer_head_kqv[(l, h)] = {}
            for proj_type, proj_vectors in this_layer_kqv.items():
                print(f'proj_type: {proj_type}, proj_vectors.shape: {proj_vectors.shape}')
                layer_head_kqv[(l, h)][proj_type] = proj_vectors[:, h, :, :]
    
    return layer_head_kqv


def convert_numpy_layer_head_kqv(layer_head_kqv):
    layer_head_kqv_np = {}
    for layer_head, kqv in layer_head_kqv.items():
        layer_head_kqv_np[layer_head] = {
            'q': kqv['q'].cpu().numpy(),
            'k': kqv['k'].cpu().numpy(),
            'v': kqv['v'].cpu().numpy(),
        }
    return layer_head_kqv_np

def do_softmax_model(layer_heads, seq_idxs, dataset):
    """
    Same logic as do_gla_model, but using the softmax model
    """
    softmax_model, softmax_tokenizer = get_softmax_attn_model_tokenizer()
    sanity_check_model(softmax_model)

    layer_head_kqv = get_attn_state_vectors(
        'keys',
        softmax_model,
        softmax_tokenizer,
        dataset,
        LAYER_HEADS,
        [4],
        max_tokens=2000
    )
    # This is a dict like so:
    # {
    #    (0, 0): { 'k': ..., 'v': ..., 'q': ... }, each of shape (batch, seq_len, head_dim)
    #    (0, 1): ...
    # }

    # Only keep the layer_head pairs that you want
    layer_head_kqv_pruned = {}
    for lh, kqv in layer_head_kqv.items():
        if lh in LAYER_HEADS:
            layer_head_kqv_pruned[lh] = kqv

    with open('softmax_layer_head_kqv.pkl', 'wb') as f:
        pickle.dump(layer_head_kqv_pruned, f)

def do_gla_model(layer_heads, seq_idxs, dataset):
    """
    Same logic as do_softmax_model, but using the gla model
    """
    gla_model, gla_tokenizer = get_linear_attn_model_tokenizer()
    sanity_check_model(gla_model)

    gla_layer_head_kqv = get_attn_state_vectors(
        'keys',
        gla_model,
        gla_tokenizer,
        dataset,
        LAYER_HEADS,
        [4],
        max_tokens=2000
    )
    # This is a dict like so:
    # {
    #    (0, 0): { 'k': ..., 'v': ..., 'q': ... }, each of shape (batch, seq_len, head_dim)
    #    (0, 1): ...
    # }
    # Only keep the layer_head pairs that you want
    gla_layer_head_kqv_pruned = {}
    for lh, kqv in gla_layer_head_kqv.items():
        if lh in LAYER_HEADS:
            gla_layer_head_kqv_pruned[lh] = kqv

    with open('gla_layer_head_kqv_pruned.pkl', 'wb') as f:
        pickle.dump(gla_layer_head_kqv_pruned, f)

if __name__ == '__main__':
    LAYER_HEADS = [(0, 0)]
    SEQ_IDXS = [4]
    dataset = get_dataset()

    do_softmax_model(LAYER_HEADS, SEQ_IDXS, dataset)