import torch
import torch.nn as nn
from timm import create_model

class ViTSmallPatch16(nn.Module):
    def __init__(self, 
                 num_classes=1000, 
                 init_type="kaiming_uniform", 
                 fc_mlp=False, 
                 hidden_dim=256, 
                 embedding_dim=384,
                 dropout=0.,
                 attn_dropout=0.,
                 mixup=None,
                 cutmix=None,
                 random_erase=None,
                 pretrained_model_path=None):
        super(ViTSmallPatch16, self).__init__()
        self.vit = create_model("vit_small_patch16_224", 
                                pretrained=False, 
                                num_classes=num_classes,  
                                proj_drop_rate=dropout, 
                                attn_drop_rate=attn_dropout)
        
        if pretrained_model_path is not None:
            self.load_partial_state_dict(pretrained_model_path)
        else:
            self.initialize_weights(init_type)
        
        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(), # currently only relu is supported
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(embedding_dim, num_classes)
 
        if init_type is not None:
            self.initialize_head(init_type)

        if mixup is not None:
            self.mixup = mixup
        else:
            self.mixup = None

        if cutmix is not None:
            self.cutmix = cutmix
        else:
            self.cutmix = None
        
        if random_erase is not None:
            self.random_erase = random_erase

    def load_partial_state_dict(self, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        # Remove the classification head weights
        state_dict.pop('vit.head.weight', None)
        state_dict.pop('vit.head.bias', None)
        self.load_state_dict(state_dict, strict=False)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def initialize_head(self, init_type):
        """
        applies initialization method on head layer
        """
        m = self.vit.head
        m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

    def reset_classifier(self, num_classes=100, fc_mlp=False, hidden_dim=256):
        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(384, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(384, num_classes)

    def forward(self, x, y=None):
        # for contrastive learning, x is a list of augmented images
        if isinstance(x, list):
            x = torch.cat(x, dim=0).to("cuda" if torch.cuda.is_available() else "cpu")
            # apply mixup, cutmix and random erase during training if enabled
        if self.training and y is not None: # only apply mixup if y is supplied
            if self.mixup is not None:
                x, y = self.mixup(x, y)
            if self.random_erase is not None: 
                x = self.random_erase(x)
            x = self.vit(x)
            return x, y
        else:
            x = self.vit(x)
            return x