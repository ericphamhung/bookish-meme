from simpler_grammar import SimpleGrammar as Grammar


# Traceback (most recent call last):
#   File "runners.py", line 476, in <module>
#     model, trmax, trmin, trmean = train(model, trl, optimizer, criterion, printevery, prepy)
#   File "runners.py", line 249, in train
#     for i, batch in enumerate(dl):
#   File "/home/jq/miniconda3/envs/generatesql/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 615, in __next__
#     batch = self.collate_fn([self.dataset[i] for i in indices])
#   File "/home/jq/miniconda3/envs/generatesql/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 615, in <listcomp>
#     batch = self.collate_fn([self.dataset[i] for i in indices])
#   File "runners.py", line 204, in __getitem__
#     lst.append(self.column_ints[db][table][yy.lower()])
# KeyError: '-'
SQL_TOKS_REM = [';', '(', ')', '-', '+']


gram = Grammar('sql_simple_transition_2.bnf')

AND_OR_TOP = 0
BASE_REP_TOP = 0

#not needed
MAX_SENT_LEN = 215

trdf = '~/spider_df_train.pkl'
tedf = '~/spider_df_test.pkl'