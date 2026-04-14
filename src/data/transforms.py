import monai.transforms as mt

def get_train_transforms():
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

def get_val_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=["image", "label"]),
        
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.EnsureTyped(keys=["image", "label"]),
        mt.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
        mt.SpatialPadd(keys=["image", "label"], spatial_size=(240, 240, 160), mode="constant"),
    ])