"""
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""


import torch
import torch.nn as nn
import torchvision.models as models



def load_resnet(model_name, pretrained=True, fmap_tensor=False, freeze=True):
    model_ft = getattr(models, model_name, "resnet101")(pretrained=pretrained)
    if fmap_tensor:
        feat_net = list(model_ft.children())[:-2]
        clf_net = list(model_ft.children())[-2:]
        clf_net = [clf_net[0]] + [Flattener()] + clf_net[1:]
        feat_net = nn.Sequential(*feat_net)
        clf_net = nn.Sequential(*clf_net)
        set_parameter_requires_grad(clf_net, freeze)
    else:
        feat_net = list(model_ft.children())[:-1]
        feat_net = nn.Sequential(*feat_net)
        clf_net = None
    set_parameter_requires_grad(feat_net, freeze)

    return feat_net, clf_net, 2048




class Flattener(nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

def set_parameter_requires_grad(model, freeze=True):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def initialize_backbone(model_name, num_classes=None, freeze=True, pretrained=True, final_clf=False, fmap_tensor=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    num_ftrs = 0
    print(f"Loading and initializing model: {model_name}")
    if  "resnet" in model_name:
        """ Resnet
        """
        input_size = 224
        if model_name == "resnet18":
            model_ft = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            model_ft = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet101":
            model_ft = models.resnet101(pretrained=pretrained)
        else:
            model_ft = models.resnet152(pretrained=pretrained)

        set_parameter_requires_grad(model_ft, freeze)

        if final_clf:
            return model_ft, input_size, 1000
            
        num_ftrs = model_ft.fc.in_features
        if num_classes is not None:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            modules = list(model_ft.children())[:-1]
            modules.append(Flattener())
            model_ft = nn.Sequential(*modules)

    elif model_name == "alexnet":
        """ Alexnet
        """
        input_size = 224
        model_ft = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, freeze)

        if final_clf:
            return model_ft, input_size, 1000

        num_ftrs = model_ft.classifier[6].in_features
        if num_classes is not None:
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        else:
            modules = list(model_ft.classifier.children())[:-1]
            modules.append(Flattener())
            model_ft.classifier = nn.Sequential(*modules)
        

    elif model_name == "vgg":
        """ VGG19_bn
        """
        input_size = 224
        model_ft = models.vgg19_bn(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, freeze)

        if final_clf:
            return model_ft, input_size, 1000

        num_ftrs = model_ft.classifier[6].in_features
        if num_classes is not None:
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        else:
            modules = list(model_ft.classifier.children())[:-2]
            modules.append(Flattener())
            model_ft.classifier = nn.Sequential(*modules)
        

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        input_size = 224
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, freeze)
        if final_clf:
            return model_ft, input_size, 1000
        num_ftrs = 512
        if num_classes is not None:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
        else:
            model_ft.classifier = Flattener()
        
    elif model_name == "densenet":
        """ Densenet
        """
        input_size = 224
        model_ft = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, freeze)
        if final_clf:
            return model_ft, input_size, 1000
        num_ftrs = model_ft.classifier.in_features
        if num_classes is not None:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.classifier = Flattener()
        

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        input_size = 299
        model_ft = models.inception_v3(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, freeze)

        if final_clf:
            return model_ft, input_size, 1000

        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        if num_classes is not None:
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
        else:
            model_ft.fc = Flattener()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_ft, input_size, num_ftrs


if __name__ == "__main__":
    
    model, in_dim, out_dim = initialize_backbone("resnet152")
    inputs = torch.rand([10, 3, 224, 224])
    output = model(inputs)
    print(out_dim)
    print(output.size())