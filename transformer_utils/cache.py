import torch


class GPTTopKCache:
    def __init__(self, k, cache_steps, tokenizer):
        """
        A cache dict, self.cache[input_ids] = [the i-th most likely token output, for i in range(k)]
        """
        self.k = k
        self.cache_steps = cache_steps
        self.tokenizer = tokenizer

        self.cache = {}

    def add(self, input_ids, output_ids, scores, beam_indices=None):
        """
        Args:
            input_ids: input sequence, which is the problem description
            output_ids: the complete generated program
            scores: scores at each generation step

        Returns:
            None
        """
        output_ids = output_ids.tolist()
        if beam_indices is not None:
            beam_indices = beam_indices.tolist()

        prefix_len = len(input_ids[0])
        # maybe do not need to cache all the way?
        output_len = min(prefix_len + self.cache_steps, len(output_ids[0]))

        for idx, end_index in enumerate(range(prefix_len, output_len)):
            for batch_idx in range(len(output_ids)):
                if beam_indices is not None:
                    beam_idx = beam_indices[batch_idx][idx]
                    # fixme really can't understand how beam_idx works, but only caching when beam_idx == batch_idx seems to work
                    if beam_idx != batch_idx: continue

                key = tuple(output_ids[batch_idx][:end_index])

                if key in self.cache.keys():
                    # already stored, possible because this prefix is expanded more than once in beam search
                    continue

                top_k_scores, top_k_tokens = torch.topk(scores[idx][batch_idx], k=self.k, sorted=True)

                # print('batch idx', batch_idx)
                # print('input', self.tokenizer.decode(key[132:]))
                # print('top k', [self.tokenizer.decode(token) for token in top_k_tokens])
                # print()

                # if key in self.cache.keys():
                #     assert top_k_tokens.tolist() == self.cache[key][0],\
                #         (self.tokenizer.decode(key[132:]), self.tokenizer.decode(top_k_tokens), self.tokenizer.decode(self.cache[key][0]))

                top_k_scores = torch.softmax(top_k_scores, dim=-1)

                self.cache[key] = (top_k_tokens.tolist(), top_k_scores.tolist())

    def get(self, input_ids):
        input_ids = tuple(input_ids)

        if input_ids in self.cache.keys():
            return self.cache[input_ids]
        else:
            return None

    def clear(self, encoded_ids=None):
        if encoded_ids is None:
            # clear cache unconditionally
            self.cache = {}
        else:
            encoded_ids = tuple(encoded_ids)
            keys_to_remove = []
            for cached_key in self.cache.keys():
                if cached_key[:len(encoded_ids)] != encoded_ids:
                    keys_to_remove.append(cached_key)
            for k in keys_to_remove: del self.cache[k]


class GPTSeqCache:
    def __init__(self):
        self.cache = {}

    def add(self, query_ids, output_ids):
        query_ids = tuple(query_ids)
        self.cache[query_ids] = output_ids

    def get(self, query_ids):
        for input_ids, output_ids in self.cache.items():
            if query_ids == output_ids[:len(query_ids)]:
                return output_ids

        return None

    def clear(self, new_state):
        self.cache = {input_ids: output_ids for input_ids, output_ids in self.cache.items() if new_state == output_ids[:len(new_state)]}
