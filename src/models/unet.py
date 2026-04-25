from monai.networks.nets import UNet

def build_unet3d() -> UNet:
    """Build and returns a 3D UNet model.

    Instantiates the 3D UNet architecture for 3 spatial dimensions, 
    4 input channels and 4 output channels. It defines a 5-level 
    feature hierarchy with specific channel sizes and strides.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4, 
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),            
        num_res_units=0,                 
    )
    return model