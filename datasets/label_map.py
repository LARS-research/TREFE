import numpy as np 
import torch 


class LabelMap(object):
    def __init__(self, character_set_file):
        self.character_set_file = character_set_file
        self.char_to_label_map = {}
        self.label_to_char_map = {}
        with open(character_set_file, 'r') as f:
            lines = f.read().split('\n')
            for i, c in enumerate(lines):
                self.label_to_char_map[i] = c 
                self.char_to_label_map[c] = i # 0 is reserved for blank required by warp_ctc
        assert 'blank' in self.char_to_label_map

    @property
    def num_classes(self):
        return len(self.char_to_label_map)

    def encode(self, text):
        if isinstance(text, str):
            label = [self.char_to_label_map[c] for c in text]
            return label
        else:
            raise TypeError("Not support this type! %s" % text)

    def label2str(self, label):
        return "".join([self.label_to_char_map[i] for i in label])

    def decode(self, ids, raw=True):
        if not isinstance(ids, np.ndarray):
            raise TypeError("indices must be np.ndarray type, but got %s" % type(ids))
        if ids.ndim == 1:
            if raw:
                chars = [self.label_to_char_map[i] for i in ids]
            else:
                chars = []
                for i, x in enumerate(ids):
                    if x == 0 or (i > 0 and x == ids[i - 1]):
                        continue
                    chars.append(self.label_to_char_map[x])
            return "".join(chars)
        else:
            words = [self.decode(i, raw=raw) for i in ids]
            return words
        
    def decode_label(self, ids, lens):
        if not (isinstance(ids, torch.Tensor) and isinstance(lens, torch.Tensor)):
            raise TypeError("ids and lens must be torch.Tensor type, but got %s" % type(ids))
        if lens.numel() == 1:
            length = lens[0]
            assert ids.numel() == length, "text with length: %d " \
                    "does not match declared length %d" % (ids.numel(), length)
            chars = [self.label_to_char_map[i.item()] for i in ids]
            return "".join(chars)
        else: # batch
            assert ids.numel() == lens.sum(), "texts with length %d " \
                    "does not match declared length %d" % (ids.numel(), lens.sum())
            texts = []
            index = 0
            for i in range(lens.numel()):
                l = lens[i]
                texts.append(self.decode_label(ids[index:index+l], torch.IntTensor([l])))
                index += l
            return texts
