from torch import nn

from pampy import match, HEAD, TAIL, _
import sys
import itertools
from utils import *
from node import Node
import numpy as np
from copy import copy

#at least colname list and table name should be given in the actual
# this is currently just for chekcing
#no longer needed. Commenting
# regex_strings = {'<<value_pattern>>':re.compile("[-+]?[0-9]+\.?[0-9]*"),
#                 '<<like_pattern>>':re.compile('\"[^\"]\"'),
#                 '<<colname_list>>':re.compile('[a-z]+'),
#                 '<<table_name>>':re.compile('[a-z]+')}









## BETTER WAY -
  ## UNKNOWN START STRINGS:
  ## http://inform7.com/sources/src/inweb/Woven/3-bnf.pdf
  ## use known
  ##
  ## KNOWN -
  ##  Just use transition matrix from each state, M
  ##  Cycles detected by M_i. (col) and nM_i. (col for nth degree) being the same
  ## do known only for now



class SimpleGrammar:

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        super(SimpleGrammar, self).__init__()


        self.gr = self.parse(file_name)
        self.gram_keys = [k for k in self.gr.keys()]
        m = len(self.gram_keys)
        self.or_mat = np.zeros((m, m))
        self.gr2 = {}
        for k in self.gram_keys:
            self.gr2[k] = self.gr[k]['items']
        self.or_mat, self.and_mat = self.get_or_and_mat()

        self.learn_ = self.learn_these()
        self.inverse_gr = {}

        for k in self.gram_keys:
            lst = self.gr[k]['items']
            assert isinstance(lst, list)
            # self.inverse_gr[string] = []
            for string in lst:
                assert isinstance(string, str)
                if string in self.inverse_gr:
                    self.inverse_gr[string].append(k)
                else:
                    self.inverse_gr[string] = [k]
        self.terminals, self.terminal_toks = [], []
        for key in self.gr.keys():
            m = ''.join(itertools.chain(*self.gr[key]['items']))
            if (grammar_tok.search(m) is None):
                self.terminals.append(key)
                mm = self.gr[key]['items']
                assert isinstance(mm, list)
                # if not isinstance(mm, list):
                #     mm = [mm]
                self.terminal_toks.append(mm[0])

        self.common_sentence = self._common_sentence(verbose=False)

    def get_or_and_mat(self):
        m = len(self.gram_keys)
        or_mat = np.zeros((m, m))
        and_mat = np.zeros((m, m))

        for j, k in enumerate(self.gram_keys):
            lst = self.gr2[k]
            cond = (self.gr[k]['type'] == 'or')

            if not cond and is_terminal(lst[0]):
                pass
            elif not cond:
                cond = (self.gr[k]['type'] == 'and')
                assert cond
                for l in lst:
                    and_mat = add_connection(and_mat, j,  \
                        self.gram_keys.index(l), True)
            else:
                for l in lst:
                    or_mat = add_connection(or_mat, j,  \
                        self.gram_keys.index(l))
        return or_mat, and_mat

    def recurse_common_sentence(self, tok, depth = 0, bef_term = False, verbose = False):
        tlist = []

        tok = self.pathone(tok)
        if verbose:
            print('{} at depth {}'.format(tok, depth))
        if bef_term:
            if tok not in self.gr:
                return None
            elif self.gr[tok]['type'] == 'term':
                return [tok]
        elif is_terminal(tok):
            return [tok]
        if self.gr[tok]['type'] == 'and':
            if verbose:
                print('{} is and'.format(tok))
            for t in self.gr[tok]['items']:
                vv = self.recurse_common_sentence(t, depth = depth + 1, bef_term=bef_term, verbose = verbose)
                if vv is not None:
                    tlist.extend(vv)
                else:
                    tlist.append(t)
        elif self.gr[tok]['type'] == 'term':
            if verbose:
                print('{} is term'.format(tok))
            tlist = self.gr[tok]['items']
        else:
            if verbose:
                print('{} is or'.format(tok))
            tlist.append(tok)

        return tlist

    def _sub_sentence(self, tok, bef_term = False, verbose = False):
        return ' '.join(self.recurse_common_sentence(tok, bef_term = bef_term, verbose=verbose))

    def _common_sentence(self, bef_term = False, verbose = False):
        return self._sub_sentence('<start>', bef_term = bef_term, verbose = verbose)


    def get_common_sentence(self):
        return self.common_sentence

    def in_(self, loc, tok):
        val = False
        print(tok)
        print(loc)
        if self.gr[loc]['type'] == 'or':
            for loc_ in self.gr[loc]['items']:
                val = val or self.in_(loc_, tok)
                if val:
                    break
        else:
            sent = self._sub_sentence(loc)
            '''
            PROBLEM IS HERE!

            '''
            fixed = False
            assert fixed
            rtok = resolve_tok.search(sent)
            if rtok is not None:
                rstr = ''
                for ii in range(len(rtok.groups())):
                    r = rtok.group(ii)
                    rstring += resolve_dict[r]
                val = re.match(rstring, sent)
            else:
                val = match_string(tok, sent)

        return val

    def which_one(self, loc, tok):
        #if tok not in self.gr
        print('{}, {}'.format(loc, tok))
        val = []
        if loc not in self.gr:
            print('loc is {}'.format(loc))
            return None
        elif self.gr[loc]['type'] == 'term':
            v = self.in_(loc, tok)
            if v:
                val.append(0)
        else:
            for i, t in enumerate(self.gr[loc]['items']):
                v = self.in_(t, tok)
                if v:
                    val.append(i)
            if self.gr[loc]['type'] == 'and':
                val = set(val)
        return val

    def which_one_rep(self, string):
        rep = []
        toks = self._tokenize(string)
        m = len(toks)
        count = 0
        com = self._common_sentence(bef_term = True)
        toks_comm = self._tokenize(com)
        traceback = []
        for i, t in enumerate(toks):

            val = self.which_one(toks_comm[count], t)
            if len(val)==0:
                count += 1
                val = self.which_one(toks_comm[count], t)
                if len(val)==0:
                    raise ValueError('{} is not a string recognized by the grammar'.format(string))
            rep.append(val)

            if len(val)>1:
                traceback.append(i)
        return rep, traceback


    def tokenized_form(self):
        lst0 = copy(self.gr['<query>']['items'])
        return lst0, copy(self.gr), copy(self.inverse_gr)


    def _match_string(self, string):
        comm = self.common_sentence.split('<')[0]
        if not match_string(comm, string):
            return False
        rest = string.replace(comm, '')



    def _tokenize(self, string):
        fn_ = fn_tok.match(string)
        string_tokens = string.split(' ')
        if fn_ is not None:
            all_tokens = []
            for tok in string_tokens:
                if fn_tok.match(tok) is not None:
                    ttok = tok.replace('(', " ")
                    ttok = ttok.replace(')', " ")
                    ttok = ttok.split(' ')
                    ttok.insert(0, '(')
                    ttok.append(')')
                    all_tokens.extend(ttok)
                else:
                    all_tokens.append(tok)

            string_tokens = rem_ws_from_list(all_tokens)
        return string_tokens

    def check_string_tokens(self, string, verbose = False):
        stringnew = copy(string)
        toks = string.strip().split(' ')
        toks = [tok.strip() for tok in toks if tok != '']
        like_in = any([s.upper()=='LIKE' for s in toks])
        from_tok = [i for (i, t) in enumerate(toks) if t.lower() == 'from'][0]
        ttoks = [(i, t) for (i, t) in enumerate(toks) if t not in self.terminal_toks]
        tt = copy(ttoks)
        whichtab = None
        for i, t in ttoks:
            if i == from_tok+1 and t in resolve_sampling['[[table_name]]']:

                whichtab = i
                table = t
        if whichtab is None:
            if verbose: print('{} does not have a known table'.format(string))
            return False, 'tab'
        ttoks = [(i, t) for i, t in ttoks if i!=whichtab]

        whichcols = []
        for i, t in ttoks:
            if isinstance(table, list) in extra_tables:
                v = resolve_sampling['[[colname_pattern]]'][table[0]]
                lsst = []
                for j, tab in enumerate(extra_tables[v]):
                    if t in resolve_sampling(['[[colname_pattern]]'])[table+j]:
                        lsst.append(j)
                        whichcols.append(i)
                if len(np.unique(lsst)) != 1:
                    raise ValueError('{}, {}'.format(lsst, stringnew, v))
            elif t in resolve_sampling['[[colname_pattern]]'][table]:
                whichcols.append(i)
        ttoks = [(i, t) for i, t in ttoks if i not in whichcols]
        if len(whichcols) == 0:
            if verbose: print('{} has no columns in string \n{}'.format(table, string))
            return False, 'col'
        ttok_repl = []
        for i, t in ttoks:
            for o in resolve_order:
                if 'like' in o and not like_in:
                    continue
                elif re.match(resolve_dict[o], t) is not None:
                    ttok_repl.append(i)
        ttoks = [(i, t) for i, t in ttoks if i not in ttok_repl]

        if len(ttoks) > 0:
            if verbose: print('Error in \n{}\nNot matched\n{}\n'.format(string, ttoks))
            return False, 'res'

        return True, None


    def new_follow(self, tok):
        tok = self.pathone(tok)
        if tok not in self.gram_keys:
            if is_terminal(tok):
                if tok in resolve_dict.keys():
                    new_val = [resolve_dict[tok]]
                else:
                    new_val = [tok + "\s"]
            else:
                raise ValueError("{} is not a terminal nor in grammar keys".format(tok))
        else:
            val = ""
            new_val = []
            if self.trans_dict[tok] == 'or':
                ind = self.gram_keys.index(tok)
                num_ = int(np.sum(self.or_mat[:, ind]))
                assert num_ > 1 and num_ == len(self.gr[tok]['items'])
                val = [val] * num_
                for ii, va in enumerate(val):
                    assert isinstance(va, str)
                    vhat = self.new_follow(self.gr[tok]['items'][ii])
                    if isinstance(vhat, str):
                        vhat = [vhat]
                    vv =  [va]*len(vhat)
                    for iii, v in enumerate(vv):
                        assert isinstance(vv, str)
                        vr = v+vhat[iii]
                        new_val.append(vr)

            elif self.trans_dict[tok] == 'and':
                ind = self.gram_keys.index(tok)
                num_ = int(np.sum(self.and_mat[:, ind]))
                assert num_ == len(self.gr[tok])
                for ii in range(num_):
                    vhat = self.new_follow(self.gr[tok]['items'][ii])
                    if isinstance(vhat, str):
                        vhat = [vhat]
                    vv =  [val]*len(vhat)
                    for iii, v in enumerate(vv):
                        assert isinstance(vv, str)
                        vr = v+vhat[iii]
                        new_val.append(vr)

            elif self.trans_dict[tok] == 'term':
                new_val = self.new_follow(self.gr[tok])
            else:
                raise ValueError("I've gotten to a place that shouldn't be possible")
        return new_val

    def in_terminal_toks(self, key):
        return key in self.terminal_toks

    def is_resolve(self, key):
        if self.in_terminal_toks(key):
            val = False
        else:
            val = test_resolve(key)
        return val



    def parse(self, fn):
        with open(fn, 'r') as f:
            text_lst = f.readlines()

        text_lst = rem_ws_from_list(text_lst)

        gram_dct = dict()
        numor = 0

        for l in text_lst:
            l = l.strip()
            m = comment.match(l)

            if m:
                val = m.start()
            else:
                val = -1

            if val == 0:
                continue
            elif val > 0:
                l = l[0:(val-1)]

            if l == "": continue

            in_parse = (begin_tok.search(l) is None)
            if in_parse:
                assert last_key in gram_dct
                gram_dct[last_key] += " "
                gram_dct[last_key] += l
            else:
                after_split = beginning_split.split(l)
                assert len(after_split) == 2
                last_key = after_split[0].strip()
                gram_dct[last_key] = after_split[1].strip()

        gram_keys = [k for k in gram_dct.keys()]

        m = len(gram_keys)

        # or_mat = np.zeros((m, m))
        # and_mat = np.zeros((m, m))
        fin_dict = dict()

        for j, k in enumerate(gram_keys):
            fin_dict[k] = dict()

            string = gram_dct[k]
            lst, cond = split_string_or(string)

            if not cond and is_terminal(lst[0]):
                fin_dict[k]['type'] = 'term'
            elif not cond:
                fin_dict[k]['type'] = 'and'
                lst, _ = split_string_and(lst[0])
            else:
                fin_dict[k]['type'] = 'or'
                # self.or_mat = add_connection(self.or_mat, j)


            if not isinstance(lst, list):
                print('{}, {}'.format(k, lst))
                assert False
            fin_dict[k]['num'] = len(lst)
            fin_dict[k]['items'] =  lst

        return fin_dict


    def get_num_learn(self):
        return len(self.learn_)
    def learn_these(self):
        dct = {}
        for k in self.gram_keys:
            if self.gr[k]['type'] == 'or':
                dct[k] = self.gr[k]['items']
        return dct

    def get_grammar_keys(self):
        return self.gram_keys

    def get_grammar(self):
        return self.gr

    def get_inverse_grammar(self):
        return self.inverse_gr

    def get_terminals(self):
        return self.terminals

    def get_terminal_toks(self):
        return self.terminal_toks


    def pathone(self, tok):
        v = tok
        while v in self.gr and self.gr[v]['num'] == 1:
            v = self.gr[v]['items'][0]
        return v

    def is_on_path(self, start_, terminal):
        #print(self.gr2[start_])
        return _item_in(self.gr2[start_], terminal)


    def from_terminal_to_token(self, tok, terminal):
        if tok in self.terminals:
            assert len(self.gr[tok]) == 1
            val = (terminal == self.gr[tok][0])
        elif tok in self.terminal_toks:
            val = (terminal == tok)
        else:
            for t in self.gr[tok]['items']:
                while (t not in self.terminal_toks) and self.gr[t]['num'] == 1:
                    t = self.gr[t]['items'][0]
                    if t in self.terminal_toks:
                        break
                val = self.from_terminal_to_token(t, terminal)
                if val > 1:
                    break
        return val

class Grammar(nn.Module):

    def __init__(self, file_name, maxdepth = -1, verbose = False):
        super(Grammar, self).__init__()
        self.loop_in = loop_in
        self.gr = {}
        self.maxdepth = maxdepth
        self.verbose = verbose

        self.num_loops = 0

        self.parse(file_name)
        mm = copy(self.and_mat)
        mmold = copy(self.and_mat)
        i = 0
        while(mm.sum()>0):
            print(mm.sum())
            print(i)
            mmold = copy(mm)
            mm = self.and_mat@mm
            if np.all(mm == mmold):
                for j, t in enumerate(mm):
                    if t.sum() > 0:
                        print('\n\n\n')
                        print(self.gram_keys[j])
                        for k, ss in enumerate(t):
                            if ss > 0:
                                print(self.gram_keys[k])
                assert False


        self.reverse_gr = {}
        for k in self.gr:
            lst = self.gr[k]
            assert isinstance(lst, list)
            for string in lst:
                assert isinstance(string, str)
                self.reverse_gr[string] = k
        self.terminals, self.terminal_toks = [], []
        for key in self.gr.keys():
            m = ''.join(itertools.chain(*self.gr[key]))
            if (grammar_tok.search(m) is None):
                self.terminals.append(key)
                mm = self.gr[key]
                if not isinstance(mm, list):
                    mm = [mm]
                self.terminal_toks.extend(mm)

    #     start_ = '<start>'
    #     # mmax = 1
    #     # for i in range(self.or_mat.shape[1]):
    #     #     mm = max(1, np.sum(self.or_mat[:, i]))
    #     #     print(mm)
    #     #     mmax *= mm
    #     # print(mmax)
    #
    #     lst = []
    #
    #     vv = copy(start_)
    #
    #     all_c = self.get_all_choices()
    #     print(all_c)
    #     print(len(all_c))
    #     assert False
    #
    #
    #
    #
    #
    #
    #     first_s = self.get_string_list(vv)
    #     print(first_s)
    #
    #     # dd = 0
    #     # stringer = '{}'
    #     # while True:
    #     #     if is_terminal_2(vv):
    #     #         if isinstance(vv, list):
    #     #             vv = ' '.join(vv)
    #     #             vv = vt
    #
    #
    # def get_choices(self, index):
    #     lst = [i for i, v in enumerate(self.or_mat[:, index].tolist()) if v !=0]
    #     # if len(lst) == 0:
    #     #     assert index == 0
    #     #     lst = [v for v in self.and_mat[:, index].tolist() if v!=0]
    #     #     assert len(lst) == 1
    #     return lst
    #
    # def get_all_choices(self):
    #     assert self.gram_keys[0] == '<start>'
    #     lst = []
    #     for i in range(self.or_mat.shape[1]):
    #         if np.sum(self.or_mat[:, i]) == 0:
    #             continue
    #         vv = self.get_choices(i)
    #         lst.append(vv)
    #     print(len(set(itertools.product(*lst))))
    #     assert False
    #     return set(itertools.product(*lst))
    #
    #
    # def get_string_list(self, tok, choice = 0):
    #     if isinstance(choice, int):
    #         choice = [choice]
    #     ss = '{}'
    #     new_tok = self.gr[tok]
    #     if self.trans_dict[tok] == 'or':
    #         new_tok = new_tok[choice[-1]]
    #         new_tok = self.get_string_list(new_tok)
    #     elif self.trans_dict[tok] == 'and':
    #         if isinstance(new_tok, list):
    #             lst = []
    #             for i in range(len(new_tok)):
    #                 lst.append(ss.format(self.get_string_list(new_tok[i])))
    #             new_tok = ' '.join(lst)
    #         else:
    #             new_tok = self.get_string_list(new_tok)
    #     else:
    #         if isinstance(new_tok, list):
    #             lst = []
    #             for i in range(len(new_tok)):
    #                 if new_tok[i] in resolve_dict:
    #                     lst.append(ss.format(resolve_dict[new_tok[i]]))
    #                 else:
    #                     lst.append(ss.format(new_tok[i]))
    #             new_tok = ' '.join(lst)
    #         else:
    #             if new_tok in resolve_dict:
    #                 new_tok = resolve_dict[new_tok]
    #
    #     return new_tok


    def sample_from_p(self, tok, p):
        pass
        # if self.trans_dict['tok'] in ['and', 'term']:
        #     return

    def follow_first(self, string):
        v = self.gr[string]
        if isinstance(v, list):
            v = v[0]
        return v



    def reverse_follow(self, string):

        if string == "":
            val = True
        elif notin(string, self.reverse_gr.keys()) and is_terminal(string):
            val = test_resolve(string) is not None
        elif notin(string, self.reverse_gr.keys()):
            val = False
        elif is_terminal(string):
            string = string.upper()
            val = self.reverse_follow(self.reverse_gr[string])
        else:
            val = True

        return val

    def in_terminal_toks(self, key):
        return key in self.terminal_toks

    def is_resolve(self, key):
        if self.in_terminal_toks(key):
            val = False
        else:
            val = test_resolve(key)
        return val


    def in_possibles(self, string, index, loc):
        if self.and_mat[:, index].sum()>0:
            assert self.or_mat[:, index].sum()==0
            lst = [i for i, v in enumerate(self.and_mat[:, index].tolist()) if v !=0]
            assert len(lst) > loc
            # if lst[]



    def match(self, string):
        fn_ = fn_tok.match(string)
        string_tokens = string.split(' ')
        if fn_ is not None:
            all_tokens = []
            for tok in string_tokens:
                if fn_tok.match(tok) is not None:
                    ttok = tok.replace('(', " ")
                    ttok = ttok.replace(')', " ")
                    ttok = ttok.split(' ')
                    ttok.insert(0, '(')
                    ttok.append(')')
                    all_tokens.extend(ttok)
                else:
                    all_tokens.append(tok)
            string_tokens = rem_ws_from_list(all_tokens)
        # print(len(string_tokens))
        # which_resolve = []
        # for i, s in enumerate(string_tokens):
        #     if self.is_resolve(s):
        #         which_resolve.append(i)
        which_resolve = [i for (i, s) in enumerate(string_tokens) if self.is_resolve(s)]

        string = '{} '*len(string_tokens)
        matched = None
        vv = '<start>'
        for i, t in enumerate(string_tokens):
            if i == 0:
                val = (t == '<select>')
            ii = self.gram_keys.index(vv)
            nn = self.and_mat[:, ii].sum()
            if nn > 0:
                if nn == 1:
                    vv = kaa



        if 0 in which_resolve:
            val = False
        else:
            val = True


        for i, t in enumerate(string_tokens):
            if not val:
                break
            if i == 0:
                val = (t == '<select>')
        for t in string_tokens:

            val = val and self.reverse_follow(t)

        #if val:

        return val

    #
    # def follow_down(self, tok, choice):
    #     if self.trans_dict[tok] = 'term':

    def parse(self, fn):
        with open(fn, 'r') as f:
            text_lst = f.readlines()

        text_lst = rem_ws_from_list(text_lst)

        gram_dct = dict()
        trans_dict = dict()
        last_key = None


        for l in text_lst:
            l = l.strip()
            m = comment.match(l)

            if m:
                val = m.start()
            else:
                val = -1

            if val == 0:
                continue
            elif val > 0:
                l = l[0:(val-1)]

            if l == "": continue

            in_parse = (begin_tok.search(l) is None)
            if in_parse:
                assert last_key in gram_dct
                gram_dct[last_key] += " "
                gram_dct[last_key] += l
            else:
                after_split = beginning_split.split(l)
                assert len(after_split) == 2
                last_key = after_split[0].strip()
                gram_dct[last_key] = after_split[1].strip()

        gram_keys = [k for k in gram_dct.keys()]

        m = len(gram_keys)

        or_mat = np.zeros((m, m))
        and_mat = np.zeros((m, m))

        for j, k in enumerate(gram_keys):
            string = gram_dct[k]
            lst, cond = split_string_or(string)

            if not cond and is_terminal(lst[0]):
                trans_dict[k] = 'term'
            elif not cond:
                trans_dict[k] = 'and'
                lst, _ = split_string_and(lst[0])
                for l in lst:
                    and_mat = add_connection(and_mat, j,  \
                        gram_keys.index(l), True)
            else:
                trans_dict[k] = 'or'
                for l in lst:
                    or_mat = add_connection(or_mat, j,  \
                        gram_keys.index(l))


            if not isinstance(lst, list):
                print('{}, {}'.format(k, lst))
                assert False
            gram_dct[k] = lst

        self.gr = gram_dct
        self.trans_dict = trans_dict
        self.or_mat = or_mat
        self.and_mat = and_mat
        self.gram_keys = gram_keys


    def get_grammar_keys(self):
        return self.gram_keys

    def get_grammar(self):
        return self.gr

    def get_reverse_grammar(self):
        return self.reverse_gr

    def get_terminals(self):
        return self.terminals

    def get_string(self, i):
        return self.all_strings[i]

    def get_num_strings(self):
        return len(self.all_strings)

    def evaluate_string(self, string):
        pass #root =

if __name__ == '__main__':
    g = Grammar('sql_simple_transition.bnf')
    print(g.match('SELECT COUNT(*) FROM thistab where i > 10 and jj <> 20 order by sd limit 10'))
    print(g.match('SELECT COUNT(*)'))
    print(g.match('suck it'))
    # for i in range(10):
    #     print(g.get_string(i))


    #gr = g.get_grammar()
    #revg = g.reverse_gr
    #print(revg['[[like_pattern]]'])
    #print(gr['<condition>'])
    #print(g.match('SELECT * FROM [[table_name]]'))

    #print(g.get_grammar_tree())
