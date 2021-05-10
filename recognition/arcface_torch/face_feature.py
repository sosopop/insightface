import torch
import torch
import os
import backbones
import cv2

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
        return self.model(img_list)

    def inference_cv(self, img_list):
        for idx, img in enumerate(img_list):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list[idx] = torch.from_numpy(img.transpose((2, 0, 1)))
        return self.inference_np(img_list)

    def inference_np(self, img_list):
        img_list = self.img_to_tensor(img_list)
        return self.inference_tensor(img_list).data.cpu().numpy()

    def similarity_np(self, feature1, feature2):
        return self.cos(torch.from_numpy(feature1), torch.from_numpy(feature2)).data.cpu().numpy()

    def similarity_tensor(self, feature1, feature2):
        return self.cos(feature1, feature2)


if __name__ == "__main__":
    face_feature = FaceFeature(
        "iresnet100", "ms1mv3_arcface_r100_fp16/backbone.pth", True)
    # face_feature = FaceFeature(
    #     "iresnet100", "ms1mv3_arcface_iresnet100/backbone.pth", False)
    # face_feature = FaceFeature(
    #     "mobilefacenetv3", "ms1mv3_arcface_mbfacenetv3/backbone.pth", False)
    # face_feature = FaceFeature(
    #     "mobilefacenet", "ms1mv3_arcface_mbfacenet/backbone.pth", True)

    # img1 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", cv2.IMREAD_COLOR)
    # img2 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0002.jpg", cv2.IMREAD_COLOR)
    # img3 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", cv2.IMREAD_COLOR)
    # import time 
    # torch.cuda.synchronize()
    # begin = time.time()
    # for i in range(10):
    #     y = face_feature.inference_cv([img1, img2, img3])
    #     sim = face_feature.similarity_np(y[0], y[1])
    #     sim = face_feature.similarity_np(y[0], y[2])
    # torch.cuda.synchronize()
    # end = time.time()
    # print(end-begin) 


    # sim = face_feature.similarity_np(y[0], y[1])
    # print(sim)
    # sim = face_feature.similarity_np(y[0], y[2])
    # print(sim)
    # sim = face_feature.similarity_np(y[1], y[2])
    # print(sim)

    img1 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0001.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Peirsol/Aaron_Peirsol_0002.jpg", cv2.IMREAD_COLOR)
    img3 = cv2.imread("/media/mengchao/dataset/feature/LFW/lfw_align_112/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", cv2.IMREAD_COLOR)
    import time 
    torch.cuda.synchronize()
    begin = time.time()
    for i in range(10):
        y = face_feature.inference_cv([img1, img2, img3])
        sim = face_feature.similarity_np(y[0], y[1])
        sim = face_feature.similarity_np(y[0], y[2])
    torch.cuda.synchronize()
    end = time.time()
    print(end-begin) 


    sim = face_feature.similarity_np(y[0], y[1])
    print(sim)
    sim = face_feature.similarity_np(y[0], y[2])
    print(sim)
    sim = face_feature.similarity_np(y[1], y[2])
    print(sim)