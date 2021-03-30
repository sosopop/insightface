import torch
import os
from backbones.mobilenetv3_export import mbv3_export

if __name__ == '__main__':
    model = mbv3_export()
    backbone_pth = os.path.join("./ms1mv3_arcface_mbfacenetv3", "backbone.pth")
    model.load_state_dict(torch.load(backbone_pth))

    # model = torch.jit.load('mobilefacenet_scripted.pt')
    # model.eval()
    # net_state_dict = model.state_dict()
    # torch.save({
    #     'iters': 0,
    #     'net_state_dict': net_state_dict},
    #     'facenet.ckpt')
        
    # model = MobileFaceNet()
    # checkpoint = torch.load('facenet.ckpt')['net_state_dict']
    # model.load_state_dict(checkpoint)

    # model = torch.load('insight-face-v3.pt')
    # net_state_dict = model.state_dict()

    #checkpoint = torch.load('BEST_checkpoint.tar')['model'].module.state_dict()
    
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, torch.randn(1, 3, 112, 112)
                    .to(torch.device("cpu")), "facenet.onnx",
                    input_names=input_names,
                    output_names=output_names)