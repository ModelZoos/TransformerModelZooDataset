import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets

train_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

        ]
    )
val_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ]
)

def get_datasets(dataset_path="/raid/dfalk/CIFAR100"):
    train_data = datasets.CIFAR100(
        root=dataset_path,
        train=True,
        download=True,
        transform=train_transform
    )
    val_data = datasets.CIFAR100(
        root=dataset_path,
        download=True,
        train=False,
        transform=val_transform
    )
    return train_data, val_data


train_set, val_set = get_datasets()

dataset = {
    "trainset": train_set,
    "testset": val_set
}

# Save the dataset to a file
torch.save(dataset, "cifar_dgx2.pt")