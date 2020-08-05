def xor(a, b):
    return (a or b) and not(a and b)
min_len = 4
nums = [str(i) for i in range(10)]
rem = ['cheese', '-', '&', '/', 'kg', "d'", 'kgs', 'la', 'x', 'des']
dont_agg_by = ['deli', 'foods', 'convenience', 'store', 'food', 'supermarket', 'farm', 'cheese']
def get_hierarchy(all_strings, all_perm = False):
    r_all = []
    for i, r in enumerate(all_strings):
        r = r.lower()
        rr = r.split(' ')

        rr = [r.replace('(', '').replace(')', '') for r in rr]
        for re in rem:
            try:
                rr.remove(re)
            except:
                pass
        rr = [s.lower() for s in rr if not any([n in s for n in nums])]
        rr = [s for s in rr if len(r)>min_len]
    
        r_all.extend(rr)
        r_all = set(r_all)
        r_all = list(r_all)

    all_rs = {}
    for r in r_all:
        rs = [s for s in all_strings if r in s]
        if len(rs) <= 1:
#             print('skipping {}'.format(r))
            continue
        string0 = rs[0]
        for i in range(1, len(rs)):
            s = SequenceMatcher(None, string0, rs[i])
            m = s.find_longest_match(0, len(string0), 0, len(rs[i]))
            string0 = string0[m[0]:(m[0]+m[2])]
            string0 = string0.strip()
        if len(string0) < len(r):
            continue
#         print('{}, {}'.format(r, string0))
        if r not in string0:
            string0 = r
        if string0 in all_rs:
            continue
        rs.sort()
        if string0=='an cream':
            continue
        if string0 in dont_agg_by:
            continue
        if len(string0) < min_len:
            continue
        all_rs[string0] = rs
    flag = False
    breaker = False
    broke_piece = None
    count = 0
    while not flag:
        
        if broke_piece is not None:
            del all_rs[broke_piece]
            broke_piece = None
#         curr_lst = list(all_rs.keys())
#         curr_lst.sort()
        to_add = {}
        count += 1
        for string0 in all_rs:

            
            if breaker:
                breaker = False
                break
            for string1 in all_rs:

                if string1 == string0:
                    continue
                set0 = set(all_rs[string0])
                set1 = set(all_rs[string1])
                set2 = set0.union(set1)
                if len(set2) == len(set1) and len(set2) == len(set0):
                    
                    maxstring = string0 if len(string0)>len(string1) else string1
                    minstring = string1 if len(string0)>len(string1) else string0
                    if ':' in maxstring and ':' not in minstring:
                        maxstring, minstring = minstring, maxstring
                    print('{0} same as {1}, deleting {1}'.format(maxstring, minstring))
                    breaker = True
                    broke_piece = minstring
                    break
                if all_perm:    
                    s0is1 = set0.intersection(set1)
                    if len(s0is1)>0:
                        to_add[string0+':'+string1] = s0is1
                else:
                    
                    if len(set2)==len(set0):
                        maxstring = string0
                        minstring = string1
                        if minstring in maxstring:
                            continue
                        print('{0} inside {1}, rearranging'.format(minstring, maxstring))
                        to_add[minstring+':'+maxstring] = list(set0)
                        breaker = True
                        broke_piece = maxstring
                        break
                    elif len(set2)==len(set1):
                        maxstring = string1
                        minstring = string0
                        if minstring in maxstring:
                            continue
                        print('{0} inside {1}, rearranging'.format(minstring, maxstring))
                        to_add[minstring+':'+maxstring] = list(set1)
                        breaker = True
                        broke_piece = maxstring
                        break
                        
        if len(to_add)>0:
            o_ = len(all_rs)
            all_rs.update(to_add)
            print('At loop {}, old len {}, new len {}'.format(count, o_, len(all_rs)))
        elif broke_piece is not None:
            print()
        else:
            flag = True
            print('Done after {}'.format(count))
        
            
    return all_rs

### get rest
prods = df['product'].unique().tolist()
prods = [p.lower() for p in prods]
prods = set(prods)
prods = list(prods)
skus = [p[6:11] for p in prods]
rest = [p[13:-25] for p in prods]