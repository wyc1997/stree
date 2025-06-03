# base strategy class


class PaddingStrat:
    def __init__(self,
                 model=None,
                 input_ids=None,
                 input_block=None,
                 output_block=None,
                 **kwargs,
    ):
        pass

    def update(self,
               model=None,
               input_ids=None,
               input_block=None,
               output_block=None,
               last_word=None,
               **kwargs):

        raise ValueError('Not implemented')
        return input_block, output_block

    def update_(self,
               model=None,
               input_ids=None,
               input_block=None,
               output_block=None,
               last_word=None,
               **kwargs):

        self.draft_ids = ['_' for _ in range(input_block.size(0))]
        raise ValueError('Not implemented: for also setting id of strat of each row / draft')
        return input_block, output_block

    def get_strat_keys_(self):
        raise ValueError('Should return list of str ids')
        return []



