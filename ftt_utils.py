"""
Utilitaires pour le FT-Transformer standard (FTT) de rtdl_lib.
Ce module fournit des fonctions supplémentaires pour étendre les fonctionnalités de FTT.
"""

import torch
from typing import Any, Dict, List, Optional, Type, Union
from rtdl_lib.modules import FTTransformer, FeatureTokenizer, Transformer


def make_baseline_with_n_heads(
    *,
    n_num_features: int,
    cat_cardinalities: Optional[List[int]],
    d_token: int,
    n_blocks: int,
    attention_n_heads: int,
    attention_dropout: float,
    ffn_d_hidden: int,
    ffn_dropout: float,
    residual_dropout: float,
    last_layer_query_idx: Union[None, List[int], slice] = None,
    kv_compression_ratio: Optional[float] = None,
    kv_compression_sharing: Optional[str] = None,
    d_out: int,
) -> 'FTTransformer':
    """Create a "baseline" `FTTransformer` with a custom number of attention heads.
    
    This is an extension of FTTransformer.make_baseline() that allows specifying
    the number of attention heads.
    
    Args:
        n_num_features: the number of continuous features
        cat_cardinalities: the cardinalities of categorical features
        d_token: the token size for each feature. Must be a multiple of attention_n_heads.
        n_blocks: the number of Transformer blocks
        attention_n_heads: the number of attention heads
        attention_dropout: the dropout for attention blocks
        ffn_d_hidden: the *input* size for the *second* linear layer in `Transformer.FFN`.
        ffn_dropout: the dropout rate after the first linear layer in `Transformer.FFN`.
        residual_dropout: the dropout rate for the output of each residual branch of
            all Transformer blocks.
        last_layer_query_idx: indices of tokens that should be processed by the last
            Transformer block.
        kv_compression_ratio: apply the technique from [wang2020linformer] to speed
            up attention modules when the number of features is large.
        kv_compression_sharing: weight sharing policy for kv_compression_ratio.
        d_out: the output size.
        
    Returns:
        FTTransformer: configured model
    """
    # Get the baseline transformer subconfig
    transformer_config = FTTransformer.get_baseline_transformer_subconfig()
    
    # Override attention_n_heads
    transformer_config['attention_n_heads'] = attention_n_heads
    
    # Set other parameters
    for arg_name in [
        'n_blocks',
        'd_token',
        'attention_dropout',
        'ffn_d_hidden',
        'ffn_dropout',
        'residual_dropout',
        'last_layer_query_idx',
        'kv_compression_ratio',
        'kv_compression_sharing',
        'd_out',
    ]:
        transformer_config[arg_name] = locals()[arg_name]
    
    # Create feature tokenizer
    feature_tokenizer = FeatureTokenizer(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_token=transformer_config['d_token'],
    )
    
    # Handle head activation for None d_out
    if transformer_config['d_out'] is None:
        transformer_config['head_activation'] = None
        
    # Handle kv compression
    if transformer_config['kv_compression_ratio'] is not None:
        transformer_config['n_tokens'] = feature_tokenizer.n_tokens + 1
    
    # Create transformer
    transformer = Transformer(**transformer_config)
    
    # Create and return FTTransformer
    return FTTransformer(feature_tokenizer, transformer)