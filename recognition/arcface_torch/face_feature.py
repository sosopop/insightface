import torch
import torch
import os
import backbones

from torch import nn
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

class FaceFeature(object):
    def __init__(self, network, checkpoint, cuda):
        self.model = self.load_model(network, checkpoint, cuda)
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def load_model(self, network, checkpoint, cuda):
        self.device = torch.device('cuda' if cuda else 'cpu')
        net = eval("backbones.{}".format(network))()
        net = net.to(self.device)

        cur_path = os.path.dirname(os.path.abspath(__file__))
        backbone_pth = os.path.join(cur_path, checkpoint)

        net.load_state_dict(torch.load(backbone_pth))
        net.eval()
        return net

    def img_to_tensor(self, img_list):
        img_tensors = torch.empty(len(img_list), 3, 112, 112)
        for idx, img in enumerate(img_list):
            img = transform(img)
            img_tensors[idx,:,:,:] = img
        return img_tensors.to(self.device)

    def inference_tensor(self, img_list):
        tensor = self.img_to_tensor(img_list)
        return self.model(tensor)

    def inference_np(self, img_list):
        tensor = self.img_to_tensor(img_list)
        return self.model(tensor).data.cpu().numpy()

    def similarity_np(self, feature1, feature2):
        return self.cos(torch.from_numpy(feature1), torch.from_numpy(feature2)).data.cpu().numpy()

    def similarity_tensor(self, feature1, feature2):
        return self.cos(feature1, feature2)

    def test(self):
        # from torchscope import scope
        # scope(self.model, input_size=(3, 112, 112))
        # x = torch.randn(2, 3, 112, 112).to(self.device)
        
        import cv2
        img1 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", cv2.IMREAD_COLOR)
        img2 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0002.jpg", cv2.IMREAD_COLOR)
        img3 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)))
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)))
        img3 = torch.from_numpy(img3.transpose((2, 0, 1)))

        y = self.inference_np([img1, img2])
        sim = self.similarity_np(y[0], y[1])
        print(sim)
        
        y = self.inference_np([img1, img3])
        sim = self.similarity_np(y[0], y[1])
        print(sim)
        
        y = self.inference_np([img2, img3])
        sim = self.similarity_np(y[0], y[1])
        print(sim)
        # y = y.cpu().detach().numpy()
        # self.similarity(y[0], y[1])


face_feature = FaceFeature(
    "mobilefacenetv3", "ms1mv3_arcface_mbfacenetv3/backbone.pth", True)
face_feature.test()
