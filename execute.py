import sys
import analysis as an
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdmolfiles import MolToSmiles

####### Settings ########
bond_file = "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/bonds_test.reax"
traj_file = "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/md.lammpstrj_test"
species_file = "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/species_test_main.out"
thermo_file =  "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/main_thermo.log"
element_list = ["C","H","Li","O"]
element_dict = {}
for i,elem in enumerate(element_list):
    element_dict[i+1] = elem
#########################

####### Parse files #######
atm_evo, smiles, smiles_id, chem_formulas = an.parse_bonds(bond_file=bond_file,
                                                           type_to_element=element_dict)

traj = np.array([frame for frame in an.parse_traj(traj_file)])
coords = traj[:,:,1:]
timesteps = list(smiles.keys())

r,_ = an.min_img_distance(coords=coords[0,:,:],traj_file=traj_file)

uniq_smiles, fa = an.first_appearance(smiles)
target_smile = "[H][C]1([H])[O][C]([O][Li])[O][C]1([H])[H]"
mol_id = smiles_id[100][66]

# Observing distance between selected molecules
"""
smiles_encoding_Li30 = []
for i in range(len(timesteps)):
    s = atm_evo["30"][i]
    smiles_encoding_Li30.append(list(uniq_smiles).index(s))

Li_id = an.get_elem_ids(traj_file=traj_file,
                        type_id=3)
O_id = an.get_elem_ids(traj_file=traj_file,
                        type_id=4)

id1 = [int(x)-1 for x in mol_id if int(x) in Li_id]
id2 = [int(x)-1 for x in mol_id if int(x) in O_id]

r_Li_O_mol66 = []
for i,t in enumerate(timesteps):
    r,_ = an.min_img_distance(coords=coords[i,:,:],traj_file=traj_file,target_time=t)
    r_Li_O_mol66.append(r[id1,id2])

r_Li_O_mol66 = np.array(r_Li_O_mol66)

plt.style.use("dark_background")
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(timesteps,r_Li_O_mol66[:,0],color='#ff00ff', linewidth=2, label=f"Li30-O141")
ax.plot(timesteps,r_Li_O_mol66[:,1],color='#ff8800', linewidth=2, label=f"Li30-O142")
ax.plot(timesteps,r_Li_O_mol66[:,2],color='#00ffff', linewidth=2, label=f"Li30-O141")
ax.set_title("Distance evolution of Li-O in mol66", color='white')
ax.set_xlabel("Timestep", color='white')
ax.set_ylabel("Distance [Å]", color='white')

twin = ax.twinx()
twin.plot(timesteps,smiles_encoding_Li30,color='#cc00ff', linewidth=2, label=f"SMILES evolution Li30")
twin.set_ylabel("Encoded SMILES", color='white')
plt.legend()
plt.grid()
plt.show()
"""

s,d,df = an.eval_species(species_file)
df["Time fs"] = df["Timestep"] * 0.1

t_dict, t_df = an.eval_thermo(thermo_file)
t_df["Time fs"] = t_df["Step"] * 0.1

########################

# Visualize species evolution
"""
cols_to_drop = ["Timestep","C","H","O"]
df.iloc[100:].drop(columns=cols_to_drop).plot(x="Time fs" ,linewidth = 2)
plt.xlabel('Timestep', fontsize = 16, fontweight = 'bold')
plt.ylabel('Count', fontsize = 16, fontweight = 'bold')
plt.title('Evolution of species over time in 300K NVT Ensemble', fontsize = 20, fontweight = 'bold')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')  # legend outside plot
plt.tight_layout()
plt.grid()
#plt.show()
"""

# Visualize thermodynamic properties
"""
t_df.plot(x="Time fs", y="PotEng",linewidth = 3, color="red")
plt.xlabel('Timestep', fontsize = 16, fontweight = 'bold')
plt.ylabel('Potential Energy [kcal/mol]', fontsize = 16, fontweight = 'bold')
plt.title('Evolution of configurational potential Energy ', fontsize = 20, fontweight = 'bold')
plt.legend(loc='upper right')  # legend outside plot
plt.tight_layout()
plt.grid()
plt.show()
"""

# Calculate and visualize radial distribution function
"""
Li_id = an.get_elem_ids(traj_file=traj_file,
                        type_id=3)
O_id = an.get_elem_ids(traj_file=traj_file,
                        type_id=4)

coord = traj[-10,:,1:]

n = [10,9,8,7,6,5,4,3,2,1]
rdf_Li_O = []
for i in n:
    coord = traj[-i,:,1:]

    dis, boxlen = an.min_img_distance(coords=coord,
                                      traj_file=traj_file,
                                      target_time=100)

    bins, rdf_Li_O_temp = an.g_r(distances=dis,
                            elem_a_id=Li_id,
                            elem_b_id=O_id,
                            boxdim=boxlen,
                            binwidth=0.025)
    rdf_Li_O.append(rdf_Li_O_temp)

rdf_Li_O = np.array(rdf_Li_O)
rdf_Li_O = np.mean(rdf_Li_O, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(bins,rdf_Li_O,linewidth=2,color="orange")
ax.set_xlabel("distance r [Å]", fontsize=14,fontweight="bold")
ax.set_ylabel("Normalized g(r)", fontsize=14,fontweight="bold")
ax.set_title("Radial Distribution  Li-O",fontsize=18,fontweight="bold")
ax.tick_params(axis="both",labelsize=14)
plt.grid(True)
plt.show()
"""

# Create Grid image of different molecular structures of chemical formula
"""
target_chem = 'C3H4LiO3'

target_smiles = set()
for ts in chem_formulas.keys():
    target_id = [idx for idx,formula in enumerate(chem_formulas[ts]) if formula == target_chem]

    for idx in target_id:
        target_smiles.add(smiles[ts][idx])

#mols = [Chem.MolFromSmiles(s) for s in target_smiles]

mols = []
for s in target_smiles:
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol:
        mol.UpdatePropertyCache(strict=False)

    mols.append(mol)

sm_legend = [str(i) for i,_ in enumerate(target_smiles)]

grid_img = Draw.MolsToGridImage(mols, molsPerRow=4,
                                subImgSize=(200, 200),
                                legends=sm_legend)

grid_img.show()
print(f"Print SMILES of Molecule #?: ")
print(f"{list(target_smiles)[int(input())]}")
"""

#"""
# generate reaction paths and occurances
timesteps = list(smiles.keys())
reaction_paths = defaultdict(int)

for i in range(len(timesteps)-1):
    t1 = timesteps[i]
    t2 = timesteps[i+1]

    atom_mapping_t1 = an.map_atoms_to_mols(smiles[t1],smiles_id[t1])
    atom_mapping_t2 = an.map_atoms_to_mols(smiles[t2], smiles_id[t2])

    pointer_t1_t2 = []
    pointer_t2_t1 = []

    # Generate list of pointers to products
    for mol in smiles_id[t1]:
        temp_pointer = set()
        for atm in mol:
            temp_pointer.add(atom_mapping_t2[atm][1])
        pointer_t1_t2.append(list(temp_pointer))

    # Generate list of pointers to reactants
    for mol in smiles_id[t2]:
        temp_pointer = set()
        for atm in mol:
            temp_pointer.add(atom_mapping_t1[atm][1])
        pointer_t2_t1.append(list(temp_pointer))

    uf = an.UnionFindReax()

    reax_ti = an.reaction_clusters(uf_object=uf,
                                   mol_list_t1=pointer_t1_t2,
                                   mol_list_t2=pointer_t2_t1)
    for n, reax in enumerate(reax_ti):
        reac = sorted([smiles[t1][i] for i in reax["reactants"]])
        prod = sorted([smiles[t2][i] for i in reax["products"]])
        if set(reac) != set(prod):
            reaction = str(reac) + " -> " + str(prod)
            reaction_paths[reaction] += 1

print(list(reaction_paths))
sorted_reaction_paths = sorted(reaction_paths.items(), key=lambda x: x[1], reverse=True)

print(f"{'Reaction':<30} {'Count':>5}")
print("-" * 36)
for reaction, count in sorted_reaction_paths:
    print(f"{reaction:<30} {count:>5}")
#"""

"""
# find when and where an element first appears
target_smile = "[H][C]1([H])[O][C]([O][Li])[O][C]1([H])[H]"

timesteps = list(smiles.keys())
t1 = timesteps[132]
t2 = timesteps[133]

# retrieve unique pathways
def map_atoms_to_mols(smiles_list, ids_list):
    atom_to_mol = {}
    for idx, ids in enumerate(ids_list):
        for atom in ids:
            atom_to_mol[atom] = (smiles_list[idx], idx)
    return atom_to_mol

sm_t1 = smiles[t1]
smid_t1 = smiles_id[t1]

sm_t2 = smiles[t2]
smid_t2 = smiles_id[t2]

atoms_t1 = map_atoms_to_mols(smiles[t1],smiles_id[t1])
#print(atoms_t1)
atoms_t2 = map_atoms_to_mols(smiles[t2], smiles_id[t2])
#print(atoms_t2)

smid_t1_t2_pointer = []
smid_t2_t1_pointer = []

for mol in smid_t1:
    pointer = set()

    for atm in mol:
        pointer.add(atoms_t2[atm][1])
    smid_t1_t2_pointer.append(list(pointer))

#print(smid_t1_t2_pointer)
sorted_smid_t1_t2_pointer = sorted(smid_t1_t2_pointer,key=len,reverse=True)
#print(sorted_smid_t1_t2_pointer)

for mol in smid_t2:
    pointer = set()

    for atm in mol:
        pointer.add(atoms_t1[atm][1])
    smid_t2_t1_pointer.append(list(pointer))

#print(smid_t2_t1_pointer)

sorted_smid_t2_t1_pointer = sorted(smid_t2_t1_pointer,key=len,reverse=True)
#print(sorted_smid_t2_t1_pointer)

set1 = [[0],     [1], [0],   [0,2], [2], [3], [4], [5], [6], [7]]
set2 = [[0,2,3], [1], [3,4], [5],   [6], [7], [8], [9]]

uf = an.UnionFindReax()

reax_ti = an.reaction_clusters(uf_object=uf,
                               mol_list_t1=smid_t1_t2_pointer,
                               mol_list_t2=smid_t2_t1_pointer)
print(reax_ti)
for n, reax in enumerate(reax_ti):
    reac = [smiles[t1][i] for i in reax["reactants"]]
    prod = [smiles[t2][i] for i in reax["products"]]
    print(f"Reaction {n}: {reac} <-> {prod}")
"""


"""
set1 = [[0],     [1], [0],   [0,2], [2], [3], [4], [5], [6], [7]]
set2 = [[0,2,3], [1], [3,4], [5],   [6], [7], [8], [9]]

def find_clusters(list1, list2):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only edges from list1[i] -> list2[j]
    for i, ptrs in enumerate(list1):
        for j in ptrs:
            union(('l1', i), ('l2', j))

    print(f"parent:{parent}")
    clusters = defaultdict(lambda: {'list1': [], 'list2': []})

    for i in range(len(list1)):
        root = find(('l1', i))
        clusters[root]['list1'].append(i)

    for j in range(len(list2)):
        # Only include list2 nodes that were actually referenced
        node = ('l2', j)
        if node in parent:
            root = find(node)
            clusters[root]['list2'].append(j)
            print(f"clusters:{clusters.values()}")

    return list(clusters.values())

clusters = find_clusters(set1, set2)
#clusters = find_clusters(smid_t1_t2_pointer, smid_t2_t1_pointer)
sys.exit()
print(f"Reactions from Timestep {t1} to {t2}")
for c in clusters:
    reac = [smiles[t1][i] for i in c['list1']]
    prod = [smiles[t2][i] for i in c['list2']]

    print(f"{reac} <-> {prod}")

    #print(f"list1{c['list1']} <-> list2{c['list2']}")

sys.exit()
"""



