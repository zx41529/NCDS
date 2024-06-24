import torch
import torch.nn as nn
import torch.nn.functional as F
import config as config
import config_muti

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR   “监督对比学习：https://arxiv.org/pdf/2004.11362.pdf.
它还支持SimCLR中的无监督对比丢失"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        if device is not None:
           self.device = device
        else:
            self.device = torch.device('cuda:'+config.cuda)      

                    

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None, 计算模型的损失。如果“labels”和“mask”都为None，
        it degenerates to SimCLR unsupervised loss:   它退化为SimCLR无监督损失：
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...]. 形状的隐藏矢量[bsz，n_views，…]。
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j  
                has the same class as sample i. Can be asymmetric.  形状[bsz，bsz]的对比掩码，mask_{i，j}=1，如果样本j与样本i具有相同的类。可以是不对称的。
        Returns:
            A loss scalar.损失标量。
        """
        
        if not features.is_cuda :
            self.device =torch.device('cpu')               

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)         #当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]  #   2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #   256 x 512
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #   256 x   512
            anchor_count = contrast_count   #   2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits  点乘
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #取点乘之后矩阵的对角线，拿出来了 
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)   #将mask 从batchsize*batchsize 变成 （anchor_count*batchsize，contrast_count*batchsize）

        # mask-out self-contrast cases
        ori_logits_mask = torch.scatter(           # 生成batch_size *2大小的对角线为0 其他均为1的方正
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        ori_mask = mask * ori_logits_mask           #消除自己和自己是同类的的影响

        logits_mask = torch.scatter(           #生成batch_size大小的对角线为0 其他均为1的方正 
            torch.ones_like(torch.rand(batch_size, batch_size).to(self.device)),
            1,
            torch.arange(batch_size ).view(-1, 1).to(self.device),
            0
        )
        logits_mask = torch.cat([logits_mask, logits_mask],dim=0)
        logits_mask = torch.cat([logits_mask, logits_mask],dim=1)


        mask = mask * logits_mask           #消除自己和自己是同类的的影响

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))   #此时log_prob[i,j] 表示i和j作为正样本，其他为负样本时 负的simclr的对比损失

        # compute mean of log-likelihood over positive
        div_mask = mask.sum(1)
        div_mask[div_mask==0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / div_mask
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

def simsiamLoss(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        # return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception




class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

'''
1                       1
    1                       1
        1
            1
                1
                    1




'''