from monai.networks.nets import SwinUNETR

def build_SwinUNETR():
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