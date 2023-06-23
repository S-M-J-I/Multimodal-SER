from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, CenterCrop, ColorJitter, RandomPerspective

# Defining the transforms:
video_frame_transform = Compose([
    ToPILImage(),
    Resize((252, 252)),
    CenterCrop((184, 184)),
    ToTensor()
])

# change frame color randomly
video_frame_augment_color = Compose([
    ToPILImage(),
    Resize((252, 252)),
    CenterCrop((184, 184)),
    ColorJitter(brightness=0.4, hue=0.3, saturation=0.4),
    ToTensor()
])

# change frame prespective randomly
video_frame_augment_persp = Compose([
    ToPILImage(),
    Resize((252, 252)),
    CenterCrop((184, 184)),
    RandomPerspective(distortion_scale=0.3, p=1.0),
    ToTensor()
])
