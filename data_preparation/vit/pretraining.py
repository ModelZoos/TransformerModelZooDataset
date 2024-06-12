import torch
from torchvision import transforms
import torchvision.datasets.imagenet as imagenet

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=1):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def get_contrastive_transforms():
    # SimCLR transformations
    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return ContrastiveTransformations(contrast_transforms, n_views=1), ContrastiveTransformations(contrast_transforms, n_views=1)


def get_transforms():
    # Supervised transformations
    supervised_transforms_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ImageNet normalization values
        ])

    supervised_transforms_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ImageNet normalization values
        ])

    return supervised_transforms_train, supervised_transforms_val


def get_datasets(training="contrastive", dataset_path="./data/"):
    if training == "contrastive":
        transforms_train, transforms_val = get_contrastive_transforms()
    elif training == "supervised":
        transforms_train, transforms_val = get_transforms()
    else:
        raise ValueError(f"Unknown transformation type {transforms}")

    train_data = imagenet.ImageNet(
        root=dataset_path,
        split="train",
        transform=transforms_train
    )

    val_data = imagenet.ImageNet(
        root=dataset_path,
        split="val",
        transform=transforms_val)

    return train_data, val_data



train_set, val_set = get_datasets("supervised", "/ds2/computer_vision/ImageNet")

dataset = {
    "trainset": train_set,
    "testset": val_set
}

# Save the dataset to a file
torch.save(dataset, "imagenet_supervised_ds2.pt")