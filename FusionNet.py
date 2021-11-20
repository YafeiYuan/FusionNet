class FusionNet(nn.Module):
    """
    Our interpretation of the feature extractor network as presented by He et al. 2019:
    "An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features"
    """

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True) # 加载resNet50预训练后的模型
        # 特征层中参数都固定住，不会发生梯度的更新
        # required_grad这一句是确定是否计算导数的。所以可以减少计算量，也就是不用求w和b的导数了，减少了计算量。只传播误差，而不计算权重和偏执的导数。
        for param in self.resnet.parameters():
            param.requires_grad = False

        # general layers without learnable parameters
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # b2 为后续融合 改变resNet50第1个block维度 初始化 例如: 7x7
        self.r2_conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.r2_conv1.apply(self.weight_init)
        self.r2_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.r2_conv2.apply(self.weight_init)
        self.r2_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.r2_conv3.apply(self.weight_init)

        # b3 为后续融合 改变resNet50第2个block维度 初始化 例如: 7x7
        self.r3_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.r3_conv.apply(self.weight_init)

        # b4 为后续融合 改变resNet50第3个block维度 初始化 例如: 7x7
        self.r4_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.r4_conv.apply(self.weight_init)

        # b5 为后续融合 改变resNet50第4个block维度 初始化 例如: 7x7
        self.r5_deconv = nn.ConvTranspose2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.r5_deconv.apply(self.weight_init)
        self.r5_conv = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        self.r5_conv.apply(self.weight_init)

        # 全连接层初始化
        self.classifier_fc1 = nn.Linear(512 * 6 * 6, 1024)  # 层数512 图片大小 6*6
        self.classifier_fc2 = nn.Linear(1024, 6)
        
        # 分类器初始化
        self.classifier_softmax = nn.LogSoftmax(dim=1)
        # self.classifier_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # get multilevel features from resnet
        r2 = self.resnet.layer1(x) 
        r3 = self.resnet.layer2(r2)
        r4 = self.resnet.layer3(r3)
        r5 = self.resnet.layer4(r4)

        # transform r2 例：(256x56x56 -> 256x14x14)
        b2 = F.relu(self.r2_conv1(r2))
        b2 = self.maxpool(b2)
        b2 = F.relu(self.r2_conv2(b2))
        b2 = self.maxpool(b2)
        b2 = F.relu(self.r2_conv3(b2))

        # transform r3 例：(512x28x28 -> 256x14x14)
        b3 = F.relu(self.r3_conv(r3))
        b3 = self.maxpool(b3)

        # transform r4 例：(1024x14x14 -> 256x14x14)
        b4 = F.relu(self.r4_conv(r4))

        # transform r5 例：(2048x7x7 -> 256x14x14)
        b5 = F.relu(self.r5_deconv(r5))
        b5 = F.relu(self.r5_conv(b5))

        # merged = torch.cat((b2, b3, b4, b5), dim=1) # 尝试4个black的不同融合
        # merged = torch.cat((b2, b5), dim=1)
        merged = torch.cat((b2, b3), dim=1)
        merged = self.flatten(merged) # 展平
        merged = self.classifier_fc1(merged) # 全连接
        merged = F.relu(merged)
        # merged = self.classifier_softmax(self.classifier_fc2(merged))

        merged = self.classifier_fc2(merged)

        return merged

    @staticmethod
    def weight_init(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
