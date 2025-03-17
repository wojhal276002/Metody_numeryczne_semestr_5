X_exact = 1.7
X_min = "10110011001100110011001"
list_X_min = [int(X_min[i]) for i in range(len(X_min))]
X_min_val = 1
for c in range(len(list_X_min)):
    X_min_val+= list_X_min[c]*2**-(c+1)
print(X_min_val)
X_pl = "10110011001100110011010"
list_X_pl = [int(X_pl[i]) for i in range(len(X_pl))]
X_pl_val = 1
for c in range(len(list_X_pl)):
    X_pl_val+= list_X_pl[c]*2**-(c+1)
print(X_pl_val)
print(f'fl(X-):{abs(X_exact-X_min_val)}','\n',f'fl(X+):{abs(X_exact-X_pl_val)}')
print(f'błąd bezwzględny = {abs(X_exact-X_pl_val)}')
blad_wzgl = abs(X_exact-X_pl_val)/abs(X_exact)
print(f'błąd względny = {blad_wzgl}')