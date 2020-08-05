from simpler_grammar import SimpleGrammar
from torch import nn
from fastai.text import RNNLearner
from fastai.data import *
from data import AugmentedDataLoader, DataBunchWithAppend

_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_1,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               Transformer: {'hid_name':'d_model', 'url':URLs.OPENAI_TRANSFORMER,
                             'config_lm':tfmer_lm_config, 'split_lm': tfmer_lm_split,
                             'config_clas':tfmer_clas_config, 'split_clas': tfmer_clas_split},
               TransformerXL: {'hid_name':'d_model',
                              'config_lm':tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
                              'config_clas':tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}

class ConditionalLearner:
    def __init__(self, filen:str):
        grammar = SimpleGrammar(filen)

        self.grammar = grammar
        self.setup_conditions()

    def setup_conditions(self):
        self.num_learn = self.grammar.get_num_learn()
        self.learn_list = []
        for i in range(self.num_learn):
            learner =


class RNNLearner_cont_or_int(RNNLearner):

    def __init__(self, data:DataBunchWithAppend, model:nn.Module, split_func:OptSplitFunc=None, clip:float=None,
                     alpha:float=2., beta:float=1., metrics=None, isint = False, **learn_kwargs):
        is_class = False
        metrics = None # ifnone(metrics, ([accuracy] if is_class else []))
        super().__init__(data, model, metrics=metrics, **learn_kwargs)
        self.callbacks.append(RNNTrainer(self, alpha=alpha, beta=beta))
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if split_func: self.split(split_func)
        self.isint = isint

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None, pbar:Optional[PBar]=None,
                  ordered:bool=False) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        self.model.reset()
        if ordered: np.random.seed(42)
        preds = super().get_preds(ds_type=ds_type, with_loss=with_loss, n_batch=n_batch, pbar=pbar)
        if ordered and hasattr(self.dl(ds_type), 'sampler'):
            np.random.seed(42)
            sampler = [i for i in self.dl(ds_type).sampler]
            reverse_sampler = np.argsort(sampler)
            preds = [p[reverse_sampler] for p in preds]
        return(preds)
