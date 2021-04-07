import torch
import torch
import os
import backbones
import cv2
import time 

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
        img_tensors = torch.empty(len(img_list), *img_list[0].size())
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
        "iresnet100", "ms1mv3_arcface_r100_fp16/backbone.pth", False)
    # face_feature = FaceFeature(
    #     "mbv3", "ms1mv3_arcface_mbfacenetv3_test/backbone.pth", False)
    # face_feature = FaceFeature(
    #     "mobilefacenet", "ms1mv3_arcface_mbfacenet/backbone.pth", True)

    # test_image_path =[
    #     "aligned_mask_0_baby1.jpg",
    #     "aligned_mask_0_baby2.jpg",
    #     "aligned_mask_0_baby3.jpg",
    #     "aligned_mask_0_huangxiaoming1.jpg",
    #     "aligned_mask_0_huangxiaoming2.jpg",
    #     "aligned_mask_0_huangxiaoming3.jpg",
    #     "aligned_mask_0_wuyanzu1.jpg",
    #     "aligned_mask_0_wuyanzu2.jpg",
    #     "aligned_mask_0_wuyanzu3.jpg",
    #     "aligned_mask_0_luhan1.jpg",
    #     "aligned_mask_0_luhan2.jpg",
    #     "aligned_mask_0_luhan3.jpg",
    #     "aligned_mask_0_yangmi1.jpg",
    #     "aligned_mask_0_yangmi2.jpg",
    #     "aligned_mask_0_yangmi3.jpg",
    #     "aligned_mask_0_fanbingbing.jpg",
    # ]

    test_image_path =[
        "aligned_0_baby1.jpg",
        "aligned_0_baby2.jpg",
        "aligned_0_baby3.jpg",
        "aligned_0_huangxiaoming1.jpg",
        "aligned_0_huangxiaoming2.jpg",
        "aligned_0_huangxiaoming3.jpg",
        "aligned_0_wuyanzu1.jpg",
        "aligned_0_wuyanzu2.jpg",
        "aligned_0_wuyanzu3.jpg",
        "aligned_0_luhan1.jpg",
        "aligned_0_luhan2.jpg",
        "aligned_0_luhan3.jpg",
        "aligned_0_yangmi1.jpg",
        "aligned_0_yangmi2.jpg",
        "aligned_0_yangmi3.jpg",
        "aligned_0_fanbingbing.jpg",
    ]
    imgs = []
    for image_path in test_image_path:
        img = cv2.imread(os.path.join("/media/mengchao/code/face/test/facedetect/output/", image_path), cv2.IMREAD_COLOR)
        imgs.append({'name': image_path, 'img': img})

    for data1 in imgs:
        sorted_map = {}
        for data2 in imgs:
            y = face_feature.inference_cv([data1['img'], data2['img']])
            sim = face_feature.similarity_np(y[0], y[1])
            sorted_map[float(sim)] = data2['name']
        sorted_map = [(k,sorted_map[k]) for k in sorted(sorted_map.keys(), reverse=True)]
        for i in range(4):
            print(sorted_map[i][1], sorted_map[i][0])
        print('---------------------------')