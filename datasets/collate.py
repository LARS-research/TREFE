import torch 


class Collate(object):

    def __call__(self, batch):
        images, labels = zip(*batch)
        lens = [len(label) for label in labels]
        b_lens = torch.IntTensor(lens)
        b_labels = torch.cat(labels)
        b_images = torch.stack(images)
        return b_images, b_labels, b_lens


class AuxCollate(object):

    def __call__(self, batch):
        images, labels = zip(*batch)
        lens = [len(label) for label in labels]
        aux_labels = torch.zeros(len(labels), max(lens))  # <eos>: 0 
        for i, label in enumerate(labels):
            aux_labels[i, :len(label)] = label 

        b_lens = torch.IntTensor(lens)
        b_labels = torch.cat(labels)
        b_images = torch.stack(images)
        return b_images, b_labels, b_lens, aux_labels.long() 