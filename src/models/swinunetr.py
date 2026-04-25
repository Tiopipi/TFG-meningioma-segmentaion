from monai.networks.nets import SwinUNETR

def build_SwinUNETR() -> SwinUNETR:
    """Build and returns a SwinUNETR model.

    Instantiates the SwinUNETR architecture for 4 input channels and
    4 output channels, a base feature size of 48, all dropout rates
    set to 0, and gradient checkpointing enable for memory efficiency.
    """
    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )
    return model