from monai.networks.nets import SegResNet


def build_segresnet() -> SegResNet:
    """Build and returns a SegResNet model.

    Instantiates the SegResNet architecture for 4 input channels and
    4 output channels, defining the number of residual blocks for the
    downsampling and upsampling paths, a base filter size of 16, and a 
    dropout probability of 0.2.
    """
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    )
    return model