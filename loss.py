import torch
import torch.nn as nn
import torch.nn.functional as F
import config as config
import config_muti

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


class SameclassConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device=None):
        super(SameclassConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        if device is not None:
           self.device = device
        else:
            self.device = torch.device('cuda:'+config.cuda)      

                    

    def forward(self, features, labels=None, mask=None):
   
        
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
            labels = labels.contiguous().view(-1, 1)         
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

        # compute logits 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        #   print (anchor_dot_contrast.size())  256 x 256

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) 
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  

        # mask-out self-contrast cases
        ori_logits_mask = torch.scatter(           
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        ori_mask = mask * ori_logits_mask           

        logits_mask = torch.scatter(          
            torch.ones_like(torch.rand(batch_size, batch_size).to(self.device)),
            1,
            torch.arange(batch_size ).view(-1, 1).to(self.device),
            0
        )
        logits_mask = torch.cat([logits_mask, logits_mask],dim=0)
        logits_mask = torch.cat([logits_mask, logits_mask],dim=1)


        mask = mask * logits_mask           

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))   

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

