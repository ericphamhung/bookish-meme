from ponyGE_src.representation import grammar as g

sqlg = 'sql_92_complete.bnf'

grm = g.Grammar(sqlg)
kk = [k for k in grm.terminals.keys()]
print(kk[24])
print(grm.terminals[kk[24]])
