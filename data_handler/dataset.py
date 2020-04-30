import numpy as np
from torch.utils.data import Dataset

from libs import encode


class IdxDataset(Dataset):

    def __init__(self, tokenizer, idx_file=None, batch_len=3200, batch_size=32, data=None, ids=None,
                 **kwargs):
        self.tokenizer = tokenizer
        self.batch_len = batch_len
        self.batch_size = batch_size
        if ids is not None:
            self.data = self.load_idx_ids(idx_file, ids)
        elif data is not None and 'idx' in data:
            self.data = data['idx']
        else:
            self.data = self.load_idx(idx_file)
        self.data = np.array(self.data)

    def load_idx(self, idx_file):
        result = []
        data_len = self.batch_len * self.batch_size
        print('loading idx', idx_file.name, 'len', data_len)
        for i, line in enumerate(idx_file):
            if i >= data_len:
                break
            result.append(tuple(map(lambda x: int(x), line.split())))
        return result

    def load_idx_ids(self, idx_file, ids):
        result = []
        ids = sorted(ids)
        print('loading idx', idx_file.name, 'len', len(ids))
        ids_idx = 0
        with open(idx_file.name, 'r') as f:
            for i, line in enumerate(idx_file):
                if i == ids[ids_idx]:
                    result.append(tuple(map(lambda x: int(x), line.split())))
        return result

    def get_loaded_length(self):
        return len(self.data)

    def __getitem__(self, item):
        idx = self.data[item]
        return idx[0], idx[1], idx[2]

    def __len__(self):
        return len(self.data),

    def get_data(self):
        return {'data': self.data}

    def split(self, partitions, eval_size=0, split_mark=('data',)):
        n = partitions
        l = len(self.data)
        param = dict(vars(self))
        param['eval_size'] = eval_size
        result = []
        data = self.get_data()
        temp_data = dict(data)
        for i in range(n):
            for k in temp_data:
                if k in split_mark:
                    temp_data[k] = data[k][i:l:n]
            param['data'] = temp_data
            result.append(type(self)(**param))
        return result


class IdxTextDataset(IdxDataset):

    def __init__(self, tokenizer, idx_file=None, sent_file=None, batch_len=100, data=None, ids=None,
                 **kwargs):
        super(IdxTextDataset, self).__init__(tokenizer, idx_file=idx_file, batch_len=batch_len, data=data,
                                             ids=ids, **kwargs)
        self.sent_data = {}
        if data is not None and 'sent' in data:
            self.sent_data = data['sent']
            all_sents = self.data[:, 2]
            if not all(x in data['sent'] for x in all_sents):
                self.load_sent(sent_file)
        else:
            self.load_sent(sent_file)

    def load_sent(self, sent_file):
        print('loading sentences', sent_file.name)
        sents = set(self.data[:, 2])
        for i in range(max(sents) + 1):
            sent = sent_file.readline()[:-1]
            if i in sents and i not in self.sent_data:
                self.sent_data[i] = encode(self.tokenizer, sent, add_eos=True, add_prefix_space=True)
        return self.sent_data

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.sent_data)

    def __getitem__(self, item):
        _, _, senti = IdxDataset.__getitem__(self, item)
        return self.sent_data[senti], senti

    def get_data(self):
        result = {'sent': self.sent_data}
        result.update(IdxDataset.get_data(self))
        return result


class IdxEntityDataset(IdxDataset):

    def __init__(self, tokenizer, idx_file=None, ent_file=None, batch_len=100, data=None, data_indexer=None, **kwargs):
        super(IdxEntityDataset, self).__init__(tokenizer, idx_file=idx_file, batch_len=batch_len, data=data,
                                               data_indexer=data_indexer, **kwargs)
        self.ent_data = {}
        if data is not None and 'ent' in data:
            self.ent_data = data['ent']
            all_ents = self.data[:, [0, 1]].reshape(1, -1).squeeze()
            if not all(x in data['ent'] for x in all_ents):
                print('Not all entity in idx exists in ent')
                self.load_ent(ent_file)
        else:
            self.load_ent(ent_file)

    def load_ent(self, ent_file):
        print('loading entities', ent_file.name)
        ents = self.data[:, [0, 1]].reshape(1, -1).squeeze()
        ents = set(ents)
        for i in range(max(ents) + 1):
            ent = ent_file.readline()[:-1]
            if i in ents and i not in self.ent_data:
                self.ent_data[i] = ent
        return self.ent_data

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.ent_data)

    def __getitem__(self, item):
        e1i, e2i, _ = IdxDataset.__getitem__(self, item)
        return self.ent_data[e1i], self.ent_data[e2i], (e1i, e2i)

    def get_data(self):
        result = {'ent': self.ent_data}
        result.update(IdxDataset.get_data(self))
        return result


class IdxFullDataset(IdxEntityDataset, IdxTextDataset):

    def __init__(self, tokenizer, idx_file=None, ent_file=None, sent_file=None, batch_len=100, data=None,
                 data_indexer=None, **kwargs):
        super(IdxFullDataset, self).__init__(tokenizer, idx_file=idx_file, ent_file=ent_file, sent_file=sent_file,
                                             batch_len=batch_len, data=data, data_indexer=data_indexer, **kwargs)

    def __getitem__(self, item):
        e1, e2, eidx = IdxEntityDataset.__getitem__(self, item)
        sent, senti = IdxTextDataset.__getitem__(self, item)
        e1, e2 = self.unify_entity(e1, sent, senti), self.unify_entity(e2, sent, senti)
        sent, senti = IdxTextDataset.__getitem__(self, item)
        return e1, e2, sent, (*eidx, senti)

    def get_loaded_length(self):
        return IdxDataset.get_loaded_length(self), len(self.sent_data), len(self.ent_data)

    def unify_entity(self, ent, sent, sent_idx):
        def in_tensor(ent, sent_tok, idx):
            tot = ''
            changed = False
            for k in range(idx, len(sent_tok)):
                temp = sent_tok[k].replace(space, '')
                if temp[:2] == '.,' and len(tot) == len(ent) - 1:
                    tot += '.'
                else:
                    tot = (tot + temp) if ent.startswith(tot + temp) else (
                        (tot + space + temp) if ent.startswith(tot + space + temp) else None)
                if tot is None:
                    return None, False
                elif tot == ent:
                    if temp[:2] == '.,':
                        sent_tok[k] = '.'
                        sent_tok.insert(k + 1, temp[1:])
                        changed = True
                    return sent_tok[idx: k + 1], changed

        space = 'Ġ'
        sent_tok = self.tokenizer.convert_ids_to_tokens(sent)
        ent_temp = ''.join(self.tokenizer.tokenize(ent))
        for i in range(len(sent_tok)):
            tmp = sent_tok[i].replace(space, '')
            if ent_temp.startswith(tmp):
                ent_tok, changed = in_tensor(ent_temp, sent_tok, i)
                if ent_tok is not None:
                    if changed:
                        self.sent_data[sent_idx] = encode(self.tokenizer, sent_tok)
                    return encode(self.tokenizer, ent_tok)
        return encode(self.tokenizer, ent)

    def get_data(self):
        result = IdxTextDataset.get_data(self)
        result.update(IdxEntityDataset.get_data(self))
        return result
