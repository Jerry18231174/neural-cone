import torch
import torch.nn as nn


class MultiresHashGrid(nn.Module):
    """
    Multi-resolution Hash Grid
    """

    index_offset = torch.tensor(
        [[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]],
        dtype=torch.int64,
        device="cuda"
    )

    big_primes = torch.tensor(
        [1, 19349663, 83492791],
        dtype=torch.int64,
        device="cuda"
    )

    def __init__(self, config: dict, bbox: torch.Tensor, twosided: bool = False) -> None:
        super(MultiresHashGrid, self).__init__()
        self.config = config
        self.bbox = bbox
        self.twosided = twosided

        self.D = config["n_features_per_level"]

        # level -> [grid_index, side, feature_dim]
        self.grids = nn.ParameterList()
        self.weights = []
        self.resolutions = []
        self.grid_sizes = []

        # Calculate grid sizes
        for i in range(config["n_levels"]):
            resolution = int(config["base_resolution"] * config["per_level_scale"] ** i)

            n_items = 0
            if ((resolution + 1) ** 3) > 2 ** config["log2_hashmap_size"]:
                n_items = 2 ** config["log2_hashmap_size"]
            else:
                n_items = (resolution + 1) ** 3
            
            grid = nn.Parameter(torch.zeros(
                (n_items, 2, self.D) if twosided else (n_items, 1, self.D),
                dtype=torch.float32,
                device="cuda"
            ))

            self.grids.append(grid)
            self.resolutions.append(resolution)
            self.grid_sizes.append(n_items)
    
    def forward(
        self,
        si_positions: torch.Tensor,
        active_side: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Interpolate grid features onto si.
        """
        pos = self._normalize_pos(si_positions)
        features = []

        for i in range(self.config["n_levels"]):
            resolution = self.resolutions[i]
            grid = self.grids[i]

            # Calculate base and offset
            # [N, 3]
            base = torch.floor(pos * resolution).to(torch.int64)
            # [N, 3]
            offset = pos * resolution - base
            comp_offset = 1 - offset

            # Calculate hash index
            # [N, 8, 3]
            index = base.unsqueeze(1) + MultiresHashGrid.index_offset
            # [N, 8]
            index = self._hash_func(index, i)
            # [8N]
            index = index.reshape(-1)

            # Calculate tri-lerp weight
            w0 = comp_offset[:, 0] * comp_offset[:, 1] * comp_offset[:, 2]
            w1 = offset[:, 0]      * comp_offset[:, 1] * comp_offset[:, 2]
            w2 = comp_offset[:, 0] * offset[:, 1]      * comp_offset[:, 2]
            w3 = offset[:, 0]      * offset[:, 1]      * comp_offset[:, 2]
            w4 = comp_offset[:, 0] * comp_offset[:, 1] * offset[:, 2]
            w5 = offset[:, 0]      * comp_offset[:, 1] * offset[:, 2]
            w6 = comp_offset[:, 0] * offset[:, 1]      * offset[:, 2]
            w7 = offset[:, 0]      * offset[:, 1]      * offset[:, 2]
            # [N, 8]
            weight = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)

            assert (index >= 0).all() and (index < self.grid_sizes[i]).all(), "Index out of range in HashGrid access!"
            # Fetch grid features
            if self.twosided:
                # [N, 8, 2, D]
                feature = grid[index].reshape(-1, 8, 2, self.D)
                # [N, 8, D]
                feature = torch.where(active_side.unsqueeze(-1).unsqueeze(-1), feature[:, :, 1], feature[:, :, 0])
            else:
                # [N, 8, D]
                feature = grid[index].reshape(-1, 8, self.D)
            # [N, D]
            feature = (feature * weight.unsqueeze(-1)).sum(dim=1)

            features.append(feature)
        
        if self.config["level_reduce"] == "Concat":
            result = torch.cat(features, dim=-1)
        elif self.config["level_reduce"] == "Mean":
            result = torch.stack(features, dim=1).mean(dim=1)
        
        return result
    
    def forward_layer_interp(
        self,
        si_positions: torch.Tensor,
        active_side: torch.Tensor = None,
        point_size: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reduce: Interpolate features between adjacent levels
        """
        pos = self._normalize_pos(si_positions, isotropic=True)
        size = point_size / (self.bbox[1] - self.bbox[0]).max()
        features = []

        sample_ratio = 0.33

        coarse_most = size > (sample_ratio / self.resolutions[0])
        fine_most = size < (sample_ratio / self.resolutions[-1])
        # print("total:", size.size())
        # print("coarse:", coarse_most.sum() / size.size(0))
        # print("fine:", fine_most.sum() / size.size(0))

        for i in range(self.config["n_levels"]):
            resolution = self.resolutions[i]
            grid = self.grids[i]

            # Calculate layer interpolation weight
            layer_weight = torch.zeros_like(size)
            if i == 0:
                upper_size, lower_size = sample_ratio / self.resolutions[0], sample_ratio / self.resolutions[1]
                valid = (size > lower_size).squeeze()
                layer_weight[coarse_most] = 1
                layer_weight[valid] = (size[valid] - lower_size) / (upper_size - lower_size)
            elif i == self.config["n_levels"] - 1:
                upper_size, lower_size = sample_ratio / self.resolutions[-2], sample_ratio / self.resolutions[-1]
                valid = (size < upper_size).squeeze()
                layer_weight[fine_most] = 1
                layer_weight[valid] = (upper_size - size[valid]) / (upper_size - lower_size)
            else:
                upper_size = sample_ratio / self.resolutions[i - 1]
                lower_size = sample_ratio / self.resolutions[i + 1]
                mid_size = sample_ratio / self.resolutions[i]
                valid_upper = ((size < upper_size) & (size > mid_size)).squeeze()
                valid_lower = ((size > lower_size) & (size < mid_size)).squeeze()
                layer_weight[valid_upper] = (upper_size - size[valid_upper]) / (upper_size - mid_size)
                layer_weight[valid_lower] = (size[valid_lower] - lower_size) / (mid_size - lower_size)

            # Calculate base and offset
            # [N, 3]
            base = torch.floor(pos * resolution).to(torch.int64)
            # [N, 3]
            offset = pos * resolution - base
            comp_offset = 1 - offset

            # Calculate hash index
            # [N, 8, 3]
            index = base.unsqueeze(1) + MultiresHashGrid.index_offset
            # [N, 8]
            index = self._hash_func(index, i)
            # [8N]
            index = index.reshape(-1)

            # Calculate tri-lerp weight
            w0 = comp_offset[:, 0] * comp_offset[:, 1] * comp_offset[:, 2]
            w1 = offset[:, 0]      * comp_offset[:, 1] * comp_offset[:, 2]
            w2 = comp_offset[:, 0] * offset[:, 1]      * comp_offset[:, 2]
            w3 = offset[:, 0]      * offset[:, 1]      * comp_offset[:, 2]
            w4 = comp_offset[:, 0] * comp_offset[:, 1] * offset[:, 2]
            w5 = offset[:, 0]      * comp_offset[:, 1] * offset[:, 2]
            w6 = comp_offset[:, 0] * offset[:, 1]      * offset[:, 2]
            w7 = offset[:, 0]      * offset[:, 1]      * offset[:, 2]
            # [N, 8]
            weight = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)

            # if not ((index >= 0).all() and (index < self.grid_sizes[i]).all()):
            #     print("index:", index.min(), index.max(), base.min(), base.max())
            #     print("grid_size:", self.grid_sizes[i])
            #     print("pos:", pos.min(dim=0).values, pos.max(dim=0).values)

            assert (index >= 0).all() and (index < self.grid_sizes[i]).all(), "Index out of range in HashGrid access!"
            # Fetch grid features
            if self.twosided:
                # [N, 8, 2, D]
                feature = grid[index].reshape(-1, 8, 2, self.D)
                # [N, 8, D]
                feature = torch.where(active_side.unsqueeze(-1).unsqueeze(-1), feature[:, :, 1], feature[:, :, 0])
            else:
                # [N, 8, D]
                feature = grid[index].reshape(-1, 8, self.D)
            # [N, D]
            feature = (feature * weight.unsqueeze(-1)).sum(dim=1)

            features.append(feature * layer_weight)
        
        result = torch.stack(features, dim=1).sum(dim=1)
        
        return result

    def _normalize_pos(self, pos: torch.Tensor, isotropic=False) -> torch.Tensor:
        """
        Normalize positions to [0, 1] range.
        """
        if isotropic:
            return (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0]).max()
        return (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
    
    def _hash_func(self, index: torch.Tensor, level: int) -> torch.Tensor:
        """
        Hash function.
        """
        resolution = int(self.config["base_resolution"] * self.config["per_level_scale"] ** level)

        if ((resolution + 1) ** 3) > 2 ** self.config["log2_hashmap_size"]:
            result = (index * MultiresHashGrid.big_primes).sum(dim=-1) % \
                     (2 ** self.config["log2_hashmap_size"])
        else:
            result = (resolution + 1) * (resolution + 1) * index[..., 0] + \
                     (resolution + 1) * index[..., 1] + \
                     index[..., 2]
        
        return result
                
        