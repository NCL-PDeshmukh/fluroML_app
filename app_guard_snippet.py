# Paste into FluroML-©PDeshmukh.py (needs only RDKit, already imported there)
from collections import defaultdict
def largest_conjugated_system(mol):
    adj = defaultdict(set)
    for b in mol.GetBonds():
        if b.GetIsConjugated() or b.GetIsAromatic():
            a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[a].add(c); adj[c].add(a)
    seen, best = set(), 0
    for s in adj:
        if s in seen: continue
        st, n = [s], 0; seen.add(s)
        while st:
            x = st.pop(); n += 1
            for y in adj[x]:
                if y not in seen: seen.add(y); st.append(y)
        best = max(best, n)
    return best

CHROMOPHORE_MIN = 8   # conjugated system smaller than 8 atoms = no chromophore (benzene ring + 1-2 atoms or smaller)

# after you compute `mol` and `prediction = model.predict(fp)[0]`:
#   if largest_conjugated_system(mol) < CHROMOPHORE_MIN:
#       prediction = 0   # show "Non-Fluorescent (no chromophore)"
