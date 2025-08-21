#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025
# Licensed under the MIT License [see LICENSE for details]

"""
Dataclass-based config for layer merging, supporting either "svd" or "slerp"
as the uniform method. In __post_init__, each LayerGroup is 'finalized' so that
it contains the definite parameters (copied from global defaults if needed).
Thus, no separate 'effective_params_for_group()' is required.

- If 'layer_merge_impl' == "svd":
    * For each group, if rank_k (or rank_v) is not specified, we fill it with the global rank_k (or rank_v).
    * We set group.slerp_t and group.slerp_gamma to None, since they're irrelevant.
- If 'layer_merge_impl' == "slerp":
    * For each group, if slerp_t (or slerp_gamma) is not specified, we fill it from the global defaults.
    * We set group.rank_k and group.rank_v to None.

layer_to_group_map is stored as '_layer_map' so that each layer index can quickly reference its group.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
from loguru import logger

@dataclass
class LayerGroup:
    """
    Represents a group of layers to be merged.
    We store both sets of parameters, but only one set is relevant
    depending on 'layer_merge_impl' in xKVConfig.

    After xKVConfig.__post_init__ runs, the 'irrelevant' fields
    for the chosen method will be set to None, while the relevant ones
    will be guaranteed to be populated (either from group overrides or global defaults).
    """
    layers: List[int] = field(default_factory=list)

    # SVD (if 'layer_merge_impl' == "svd")
    rank_k: Optional[int] = None
    rank_v: Optional[int] = None

    # SLERP (if 'layer_merge_impl' == "slerp")
    slerp_t: Optional[float] = None
    slerp_gamma: Optional[float] = None

    def __post_init__(self):
        if not self.layers:
            raise ValueError("LayerGroup must have at least one layer index.")


@dataclass
class xKVConfig:
    """
    Stores a single merging method: 'svd' or 'slerp'.

    - If 'svd': We have global defaults rank_k / rank_v. Each group
      can override (group.rank_k / group.rank_v), or we fill them from the global.
    - If 'slerp': We have global defaults slerp_t / slerp_gamma. Each group
      can override (group.slerp_t / group.slerp_gamma), or we fill them from the global.

    We finalize each group's relevant parameters in __post_init__, removing the need
    for a separate "effective_params_for_group()" method.
    """
    # Optional total number of layers for validation
    num_layers: Optional[int] = None
    
    layer_merge_impl: str = "svd"  # "svd" or "slerp"

    # Global SVD defaults
    rank_k: Optional[int] = None
    rank_v: Optional[int] = None

    # Global SLERP defaults
    slerp_t: float = 0.5
    slerp_gamma: float = 1.0

    merge_key: bool = True
    merge_value: bool = True

    # Quantizer configuration
    kv_bits: int = 16  # KV cache bit width, 16 means no quantization
    group_size: int = 0  # 0 stands for per-token quant
    sym: bool = False  # Symmetric or Asymmetric
    clip_ratio: float = 1.0
    hadamard: bool = True
    

    # All groups
    layer_groups: List[LayerGroup] = field(default_factory=list)

    # Catch-all for future expansions
    extra_kwargs: dict = field(default_factory=dict)

    # We'll store the layer->group map here (built once in __post_init__)
    _layer_map: Dict[int, LayerGroup] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Validate the method
        if self.layer_merge_impl not in ("svd", "slerp"):
            raise ValueError(
                f"Invalid layer_merge_impl '{self.layer_merge_impl}'. "
                "Must be 'svd' or 'slerp'."
            )

        # 1) Finalize each group's parameters
        if self.layer_merge_impl == "svd":
            # Make sure each group has rank_k, rank_v set; clear slerp fields
            for grp in self.layer_groups:
                # Fill in rank_k, rank_v if not specified
                grp.rank_k = grp.rank_k if grp.rank_k is not None else self.rank_k
                grp.rank_v = grp.rank_v if grp.rank_v is not None else self.rank_v
                # Nullify irrelevant fields
                grp.slerp_t = None
                grp.slerp_gamma = None

        else:  # "slerp"
            # Make sure each group has slerp_t, slerp_gamma set; clear rank_k/rank_v
            for grp in self.layer_groups:
                grp.slerp_t = grp.slerp_t if grp.slerp_t is not None else self.slerp_t
                grp.slerp_gamma = grp.slerp_gamma if grp.slerp_gamma is not None else self.slerp_gamma
                # Nullify irrelevant fields
                grp.rank_k = None
                grp.rank_v = None

        # 2) Build layer->group map once
        self._layer_map = self._build_layer_to_group_map(raise_if_duplicate=True)

        # 3) If num_layers is set, validate no group references a layer_idx >= num_layers
        if self.num_layers is not None:
            self._validate_num_layers()
    
    def _validate_num_layers(self):
        """
        Ensure that for every layer in every group, layer_idx < num_layers.
        """
        for grp in self.layer_groups:
            for lyr in grp.layers:
                if lyr >= self.num_layers:
                    raise ValueError(
                        f"Group has a layer index {lyr} which exceeds "
                        f"the declared num_layers={self.num_layers} (max index {self.num_layers - 1})."
                    )
    
    def _build_layer_to_group_map(self, raise_if_duplicate=True) -> Dict[int, LayerGroup]:
        """
        Internal method to build {layer_idx -> LayerGroup}.
        Raises if a layer is found in multiple groups (and raise_if_duplicate=True).
        """
        layer_map: Dict[int, LayerGroup] = {}
        for grp in self.layer_groups:
            for lyr in grp.layers:
                if lyr in layer_map and raise_if_duplicate:
                    raise ValueError(
                        f"Layer {lyr} appears in multiple groups: "
                        f"{layer_map[lyr]} and {grp}"
                    )
                layer_map[lyr] = grp
        return layer_map

    def get_group_for_layer(self, layer_idx: int) -> Optional[LayerGroup]:
        """
        Return the LayerGroup that 'layer_idx' belongs to, or None if not found.
        Because we've finalized each group's parameters, you can directly read
        group.rank_k or group.slerp_t, etc.
        """
        return self._layer_map.get(layer_idx, None)

    @classmethod
    def from_yaml(cls, path: str) -> "xKVConfig":
        """
        Load config from a YAML file, e.g.:

          xKV_config:
            num_layers: 12
            layer_merge_impl: "svd"
            rank_k: 128
            rank_v: 64
            slerp_t: 0.8
            slerp_gamma: 0.05
            merge_key: true
            merge_value: true
            layer_groups:
              - layers: [0,1]
                rank_k: 256
              - layers: [2,3]
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        conf_data = raw.get("xKV_config", {})
        group_data = conf_data.pop("layer_groups", [])

        # Build LayerGroup objects
        groups = [LayerGroup(**gd) for gd in group_data]
        return cls(layer_groups=groups, **conf_data)

    def to_dict(self) -> dict:
        """
        Return a dict of top-level fields, excluding layer_groups.
        We'll handle layer_groups in to_yaml().
        """
        d = {
            "num_layers": self.num_layers,
            "layer_merge_impl": self.layer_merge_impl,
            "rank_k": self.rank_k,
            "rank_v": self.rank_v,
            "slerp_t": self.slerp_t,
            "slerp_gamma": self.slerp_gamma,
            "merge_key": self.merge_key,
            "merge_value": self.merge_value,
            "kv_bits": self.kv_bits,
            "group_size": self.group_size,
            "sym": self.sym,
            "clip_ratio": self.clip_ratio,
            "hadamard": self.hadamard,
        }
        d.update(self.extra_kwargs)
        return d

    def to_yaml(self, path: str):
        """
        Dump this config to a YAML file under the key 'xKV_config'.
        We'll just store each group's current fields (including any finalization).
        """
        data = self.to_dict()

        group_list = []
        for grp in self.layer_groups:
            gd = {"layers": grp.layers}
            if grp.rank_k is not None:
                gd["rank_k"] = grp.rank_k
            if grp.rank_v is not None:
                gd["rank_v"] = grp.rank_v
            if grp.slerp_t is not None:
                gd["slerp_t"] = grp.slerp_t
            if grp.slerp_gamma is not None:
                gd["slerp_gamma"] = grp.slerp_gamma
            group_list.append(gd)

        data["layer_groups"] = group_list
        with open(path, "w") as f:
            yaml.safe_dump({"xKV_config": data}, f, sort_keys=False)

    def __str__(self) -> str:
        """
        Custom multi-line repr for more readable debugging.
        """
        lines = []
        lines.append(f"{self.__class__.__name__}(")
        lines.append(f"  # Global params:")
        lines.append(f"  num_layers={self.num_layers},")
        lines.append(f"  layer_merge_impl={self.layer_merge_impl!r},")
        lines.append(f"  rank_k={self.rank_k!r}, rank_v={self.rank_v!r},")
        lines.append(f"  slerp_t={self.slerp_t!r}, slerp_gamma={self.slerp_gamma!r},")
        lines.append(f"  merge_key={self.merge_key}, merge_value={self.merge_value},")
        lines.append(f"  # Quantizer params:")
        lines.append(f"  kv_bits={self.kv_bits}, group_size={self.group_size},")
        lines.append(f"  sym={self.sym}, clip_ratio={self.clip_ratio}, hadamard={self.hadamard},")
        lines.append(f"  # {len(self.layer_groups)} groups:")
        for idx, grp in enumerate(self.layer_groups):
            grp_repr = repr(grp).replace("\n", "\n    ")
            lines.append(f"    [{idx}] -> {grp_repr}")
        lines.append(")")
        return "\n".join(lines)


# --------------------- Utility Functions --------------------- #
def generate_consecutive_layer_groups(
    start_layer: int,
    end_layer: int,
    group_size: int,
    stride: int = 1,
) -> List[LayerGroup]:
    """
    Chunk layers from [start_layer..end_layer] into consecutive groups
    of size 'group_size'. For example, if start=0, end=5, group_size=2,
    stride=1, you get groups [0,1], [2,3], [4,5]. If start=3, end=15,
    group_size=2, stride=4, you get groups [3,7], [11,15].
    
    By default, rank_k/rank_v/slerp_t/slerp_gamma are None here.
    They will be filled (or remain None) during xKVConfig.__post_init__.
    """
    groups = []
    current = start_layer
    while current <= end_layer:
        grp_end = min(current + group_size * stride - 1, end_layer)
        groups.append(LayerGroup(layers=list(range(current, grp_end + 1, stride))))
        current = grp_end + 1
    return groups


def generate_consecutive_xKV_config(
    layer_merge_impl: str = "svd",
    start_layer: int = 0,
    end_layer: int = 31,
    num_layers: Optional[int] = None,
    group_size: int = 2,
    rank_k: Optional[int] = 256,
    rank_v: Optional[int] = 768,
    slerp_t: float = 0.5,
    slerp_gamma: float = 1.0,
    merge_key: bool = True,
    merge_value: bool = True,
    # Quantization parameters
    kv_bits: int = 16,
    quantizer_group_size: int = 0,
    sym: bool = False,
    clip_ratio: float = 1.0,
    hadamard: bool = True,
    extra_kwargs: dict = None,
) -> xKVConfig:
    """
    Quickly build a xKVConfig with consecutive-layer groups.
    
    Args:
      layer_merge_impl: "svd" or "slerp"
      start_layer: First layer index
      end_layer: Last layer index (inclusive)
      group_size: Number of layers per group
      rank_k, rank_v: Global SVD defaults (if 'svd' method)
      slerp_t, slerp_gamma: Global SLERP defaults (if 'slerp' method)
      merge_key, merge_value: Whether to apply merges to key/value
      kv_bits: KV cache bit width, 16 means no quantization
      quantizer_group_size: Group size for quantization, 0 means per-token quantization
      sym: Use symmetric quantization
      clip_ratio: Clip ratio for quantization
      hadamard: Use Hadamard transform for quantization
      extra_kwargs: Additional fields stored in the config

    Returns:
      A fully constructed xKVConfig with consecutive groups,
      each group covering a chunk of [start_layer..end_layer].
    """
    if end_layer == -1:
        assert num_layers is not None, "Must provide num_layers if end_layer is -1."
        logger.info(f"End layer not specified, using num_layer={num_layers} - 1.")
        end_layer = num_layers - 1
    layer_groups = generate_consecutive_layer_groups(start_layer, end_layer, group_size)
    return xKVConfig(
        num_layers=num_layers,
        layer_merge_impl=layer_merge_impl,
        rank_k=rank_k,
        rank_v=rank_v,
        slerp_t=slerp_t,
        slerp_gamma=slerp_gamma,
        merge_key=merge_key,
        merge_value=merge_value,
        # Quantization parameters
        kv_bits=kv_bits,
        group_size=quantizer_group_size,
        sym=sym,
        clip_ratio=clip_ratio,
        hadamard=hadamard,
        layer_groups=layer_groups,
        extra_kwargs=extra_kwargs or {}
    )


# ------------------ Example Usage ------------------ #
if __name__ == "__main__":

    # # Example SVD config
    # g1 = LayerGroup(layers=[0,1], rank_k=256, rank_v=64)
    # g2 = LayerGroup(layers=[2,3])  # fallback
    # g3 = LayerGroup(layers=[4,5], rank_k=64)  # partial override
    # svd_cfg = xKVConfig(
    #     layer_merge_impl="svd",
    #     rank_k=128, rank_v=32,   # global fallback
    #     slerp_t=0.9, slerp_gamma=0.1,  # irrelevant
    #     merge_key=True, merge_value=True,
    #     layer_groups=[g1,g2,g3],
    # )

    # print(svd_cfg)

    # # Let's see final group info
    # for i, grp in enumerate(svd_cfg.layer_groups):
    #     print(f"SVD Group {i} => layers={grp.layers}, rank_k={grp.rank_k}, rank_v={grp.rank_v}, "
    #           f"slerp_t={grp.slerp_t}, slerp_gamma={grp.slerp_gamma}")

    # # Write + read YAML
    # svd_yaml = "svd_config.yaml"
    # svd_cfg.to_yaml(svd_yaml)
    # loaded_svd = xKVConfig.from_yaml(svd_yaml)
    # print("\n[SVD] After load, let's see group for layer=4:")
    # group_for_4 = loaded_svd.get_group_for_layer(4)
    # print(f"  group.layers={group_for_4.layers}, rank_k={group_for_4.rank_k}, rank_v={group_for_4.rank_v}")

    # print("\n" + "="*60 + "\n")

    # # Example SLERP config
    # g4 = LayerGroup(layers=[0,1], slerp_t=0.95)
    # g5 = LayerGroup(layers=[2,3,4], slerp_gamma=0.2)
    # slerp_cfg = xKVConfig(
    #     layer_merge_impl="slerp",
    #     rank_k=999, rank_v=888,  # irrelevant
    #     slerp_t=0.75, slerp_gamma=0.05,
    #     merge_key=True, merge_value=True,
    #     layer_groups=[g4,g5],
    # )

    # for i, grp in enumerate(slerp_cfg.layer_groups):
    #     print(f"SLERP Group {i} => layers={grp.layers}, slerp_t={grp.slerp_t}, "
    #           f"slerp_gamma={grp.slerp_gamma}, rank_k={grp.rank_k}, rank_v={grp.rank_v}")

    # # Write + read YAML
    # slerp_yaml = "slerp_config.yaml"
    # slerp_cfg.to_yaml(slerp_yaml)
    # loaded_slerp = xKVConfig.from_yaml(slerp_yaml)
    # print("\n[SLERP] After load, let's see group for layer=3:")
    # group_for_3 = loaded_slerp.get_group_for_layer(3)
    # print(f"  group.layers={group_for_3.layers}, slerp_t={group_for_3.slerp_t}, slerp_gamma={group_for_3.slerp_gamma}")

    
    
    
    # Customized config
    # 1) Generate a consecutive SVD config with groups of size 2 (e.g. [0,1], [2,3], [4,5], etc.)
    cfg_svd = generate_consecutive_xKV_config(
        layer_merge_impl="svd",
        start_layer=0,
        end_layer=31,
        group_size=2,
        rank_k=128,
        rank_v=128,
    )
    cfg_svd.to_yaml("svd_consecutive_u_g2_k128_v128.yaml")
    print("\n[SVD consecutive config]\n", cfg_svd)