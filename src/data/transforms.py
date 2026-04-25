import monai.transforms as mt

def get_train_transforms() -> mt.Compose:
    """Generates and returns the sequence of transforamtions for the training dataset.
    
    Applies a series of transforms that prepare the data and includes data augmentation 
    techniques such as random spatial crops and random flips.

    Returns:
        A MONAI compose object with the training transformation pipeline.
    """
    return mt.Compose([
        mt.LoadImaged(keys=["image", "label"]),
        
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.EnsureTyped(keys=["image", "label"]),
        mt.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
        
        mt.RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
        
        mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ])

def get_val_transforms() -> mt.Compose:
    """Generates and returns the sequence of transforamtions for the validation dataset
    
    Applies transformrs to prepare the validation data. It also applies saptial padding to
    ensure all volumes have the same size (240, 240, 160). 

    Returns:
        A MONAI compose object with the validation transformation pipeline.
    """
    return mt.Compose([
        mt.LoadImaged(keys=["image", "label"]),
        
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.EnsureTyped(keys=["image", "label"]),
        mt.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
        mt.SpatialPadd(keys=["image", "label"], spatial_size=(240, 240, 160), mode="constant"),
    ])