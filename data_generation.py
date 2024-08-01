import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor


def data_generator(
        n_train,
        n_test,
        encode_input=True,
        encode_output=True,
):
    x_train = torch.randn(n_train, 1, 16, 16)
    y_train = x_train ** 2 + 2
    x_test = torch.randn(n_test, 1, 16, 16)
    y_test = x_test ** 2 + 2

    # %% 归一化
    encoding = "channel-wise"
    # encode_input = True
    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        x_train = input_encoder.transform(x_train)
        x_test = input_encoder.transform(x_test)
    else:
        input_encoder = None

    sample_index = 0
    x_sample = x_train[sample_index].squeeze().numpy()  # 去除多余的维度以便于绘图

    # encode_output = True
    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None

    #%%
    # 训练
    batch_size = n_train
    train_db = TensorDataset(
        x_train,
        y_train
    )
    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    # 测试
    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_batch_size = n_test
    test_loader = DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    train_resolution = 32
    test_loaders = {train_resolution: test_loader}


    # %%
    grid_boundaries = [[0, 1], [0, 1]]
    positional_encoding = True
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    return train_loader, test_loaders, data_processor

# %%
# # 绘制图形
# plt.figure(figsize=(5, 5))
# plt.imshow(x_sample, cmap='viridis')
# plt.colorbar()
# plt.title('x_train Sample')
# plt.show()
