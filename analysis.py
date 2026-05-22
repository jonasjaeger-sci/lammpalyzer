"""
This script contains the methods to analyze the output files of the lammps simulations with reaxFF
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdmolfiles import MolToSmiles
from file_read_backwards import FileReadBackwards


def eval_species(species_file):
    """
    Function evaluate species file from lammps:
    - list of unique species
    - number of respective species over time
    Parameters
    ----------
    species_file: string
        the name/location of the species file

    Returns
    -------

    """

    # extract list of unique elements
    species_set = set()
    static_cols = ["Timestep", "No_Moles", "No_Specs"]

    with open(species_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                species_set.update(c for c in cols if c not in static_cols)

    # initialize dictionary with list as keys
    species_set = sorted(species_set)
    species_dict = {"Timestep": []}
    for species in species_set:
        species_dict[species] = []
    #print(f"dict: \n{species_dict}")

    # read each line and track how many instances of a species are present
    with open(species_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_cols = line.lstrip("#").split()
            else:
                vals = line.split()
                current_row = dict(zip(current_cols, vals))
                species_dict["Timestep"].append(int(current_row["Timestep"]))
                for species in species_set:
                    species_dict[species].append(int(current_row.get(species,0)))

    species_df = pd.DataFrame(species_dict)

    return species_set, species_dict, species_df

"""
filepath = "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/species_test_main.out"
s,d,df = eval_species(filepath)

df.iloc[1:].drop(columns="Timestep").plot(linewidth = 3)
plt.xlabel('Timestep', fontsize = 16, fontweight = 'bold')
plt.ylabel('Count', fontsize = 16, fontweight = 'bold')
plt.title('Species over time', fontsize = 20, fontweight = 'bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
plt.tight_layout()
plt.grid()
plt.show()
"""

def eval_thermo(thermo_file, indicator1="Step", indicator2="Loop"):
    """
    function to evaluate thermodynamic properties file
    Parameters
    ----------
    thermo_file: string
        name/location of the thermodynamic properties file
    Returns
    -------

    """
    # initialize dictionary
    thermo_dict = {}
    switch = 0

    # get column headers as list and read each line value
    with open(thermo_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith(indicator1):
                thermo_cols = line.split()
                switch = 1

                for col in thermo_cols:
                    thermo_dict[col] = []

                continue

            if line.startswith(indicator2):
                switch = 0
                break

            if switch == 1:
                vals = line.split()
                for val, col in zip(vals, thermo_cols):
                    thermo_dict[col].append(float(val))

    thermo_df = pd.DataFrame(thermo_dict)

    return thermo_dict, thermo_df

"""
thermo_path = "/home/jonas/lammps/projects/lithium_elyte/EC_Li_60_40/main_thermo.log"
t_dict, t_df = eval_thermo(thermo_path)

t_df["Temp"].plot(linewidth = 3, color="red")
plt.xlabel('Step', fontsize = 16, fontweight = 'bold')
plt.ylabel('Temperature [K]', fontsize = 16, fontweight = 'bold')
plt.title('Species over time', fontsize = 20, fontweight = 'bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
plt.tight_layout()
plt.grid()
plt.show()
"""

class UnionFind():

    def __init__(self):
        self.root = {}
        self.rank = {}

    def add(self,val):
        """
        add a value and give it an initial rank
        Parameters
        ----------
        val: int
            value that is to be added, in this case the atomic ID

        Returns
        -------

        """
        if val not in self.root: # check if value is not already added
            self.root[val] = val
            self.rank[val] = 0

    def find(self,val):
        """
        recursive function to find the root with the highest rank of the given value
        Parameters
        ----------
        val: int
            the value whose root is being searched for

        Returns
        -------

        """
        if self.root[val] != val:
            self.root[val] = self.find(self.root[val])
        return self.root[val]

    def union(self,val1,val2):
        """
        a function to unite the two values and link them to the root with the highest rank

        Parameters
        ----------
        val1: int
            first edgepoint, the ID of the main atom in the lammps row
        val2: int
            second edgepoint, the ID of the bonded atom in the lammps row

        Returns
        -------
        """
        # get the root of both values
        root_val1, root_val2 = self.find(val1), self.find(val2)

        # if both have the same root, the already belong to the same set/cluster -> skip
        if root_val1 == root_val2:
            return

        # unite the values by linking them to the one with the higher rank
        if self.rank[root_val1] < self.rank[root_val2]:
            root_val1, root_val2 = root_val2, root_val1
        self.root[root_val2] = root_val1

        if self.rank[root_val1] == self.rank[root_val2]:
            self.rank[root_val1] += 1

    def cluster(self):
        """
        function to return a list of connected cluster representing the molecules

        Returns
        -------
        """
        cluster = defaultdict(set)

        for val in self.root:
            cluster[self.find(val)].add(val)
        return list(cluster.values())

def eval_bonds(bond_file,elem_list):
    """
    function to parse the bond file using the UnionFind class to obtain the
    clusters/molecule per timestep
    Parameters
    ----------
    bond_file: string
        name/location of the bond file
    elem_list: list of strings
        ordered list of the elements comprising the system. Order must be the same
        as in the lammps system (-> .data file)

    Returns
    -------
    cluster_dict: dictionary
        dictionary referring to the constituent atom ID for each molecule at each timestep
        -> Keys: timestep - Values: list containing lists of ElementIDs for each molecule
    """
    elem_dict = {}
    for type_id,type in enumerate(elem_list):
        elem_dict[type_id+1] = type

    cluster_dict = {}
    current_time = None

    with open(bond_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                if "Timestep" in line:

                    # save first until N-1 timestep blocks
                    if current_time is not None: # if there was a prior timestep save dict entry
                        cluster_dict[current_time] = uf.cluster()

                    parts = line.split()
                    current_time = int(parts[parts.index("Timestep") + 1])
                    uf = UnionFind()

                continue

            # obtain main atom ID and # of bonds
            vals = line.split()

            if len(vals) < 3:
                continue # to prevent error from last (empty) line

            main_id = int(vals[0])
            n_bonds = int(vals[2])

            # Union-Find-Workflow
            uf.add(main_id)

            for i in range(n_bonds):
                bonded_id = int(vals[3 + i])
                uf.add(bonded_id)
                uf.union(main_id, bonded_id)

    # save last timestep blocks
    if current_time is not None:
        cluster_dict[current_time] = uf.cluster()

    # generate dictionary of which atomic index corresponds to which element
    ind_elem_dict = {}
    with open(bond_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            vals = line.split()
            # check if key already exists - if so exit loop
            if vals[0] in ind_elem_dict:
                break

            ind_elem_dict[vals[0]] = elem_dict[int(vals[1])]

    # transform atomic indexes into element descriptors in cluster_dict
    timesteps = cluster_dict.keys()
    cluster_elements_dict = {}
    for ts in timesteps:
        cluster_elements_dict[ts] = [list(s) for s in cluster_dict[ts]]

        for cluster in range(len(cluster_elements_dict[ts])):
            cluster_elements_dict[ts][cluster] = [ind_elem_dict[str(idx)] for idx in cluster_elements_dict[ts][cluster]]

    return cluster_dict


def bo_to_rdkit_bond(bo):
    """
    function to determine the rdkit bond type from the bodn order
    Parameters
    ----------
    bo: float
        calculated continuous bond order value from LAMMPS bond file

    Returns
    -------
    Chem.BondType
    """
    if bo >= 2.5:
        return Chem.BondType.TRIPLE
    if bo >= 1.5:
        return Chem.BondType.DOUBLE
    #if bo >= 1.49:
    #    return Chem.BondType.AROMATIC
    else:
        return Chem.BondType.SINGLE

def parse_bonds(bond_file,type_to_element):
    """

    Parameters
    ----------
    bond_file: string
        path of the bond file
    type_to_element: dict
        dictionary of the elements identical to the one used for LAMMPS system
        -> Keys: ElementID - Value: Element

    Returns
    -------
    atom_evolution: dict
        dictionary of how each Element develops in the system and to which molecule it belongs
        -> keys: ElementID - Value: list of consecutive SMILES notation for each timestep
    smiles: dict
        dictionary containing the smiles notations of every molecule present at each timestep
        -> keys: timestep - Value:list of smiles notations present at the very timestep
    smiles_atoms: dict
        dictionary containing the ElementID of the respective SMILES notation in the smiles dict
        -> keys: timestep - Value:list of ElementID of smiles notations present at the very timestep
    chem_formulas: dict
        dictionary containing the basic chemical formulas of every molecule present at each timestep
        -> keys: timestep - Value:list of chemical formulas at the very timestep
    """

    atoms = {}
    bonds = []
    atom_evolution = defaultdict(list)
    smiles = {}
    smiles_atoms = {}
    chem_formulas = {}
    counter = 0
    mol = Chem.RWMol()

    with open(bond_file) as f:
        n_lines = len(f.readlines())

    with open(bond_file) as f:
        for line in tqdm(f, total = n_lines, desc = "Parsing bond file"):
            line = line.strip()

            if line.startswith("#"):
                if "Timestep" in line:
                    timestep = int(line.split()[-1])

                if "Number of particles" in line:
                    n_atoms = int(line.split()[-1])

                continue

            parts = line.split()

            if len(parts) < 3:
                break
            atoms[parts[0]] = type_to_element[int(parts[1])]

            atom_i = int(parts[0])
            n_bonds = int(parts[2])
            bonded_atoms = [int(x) for x in parts[3:3+n_bonds]]
            bond_orders = [float(x) for x in parts[4+n_bonds:4*2*n_bonds]]
            for atom_j, bo in zip(bonded_atoms, bond_orders):
                if atom_i < atom_j: # avoid duplicate bonds
                    bonds.append([atom_i,atom_j,bo_to_rdkit_bond(bo)])

            counter += 1 # count until every particle is registered

            # construct mol object for rdkit if all atoms are registered
            if counter == n_atoms:
                atom_list = list(atoms.keys())

                # add atoms:
                for atom in atoms.values():
                    a = Chem.Atom(atom)
                    a.SetNoImplicit(True) # suppressing implicit valence Hydrogen filling
                    mol.AddAtom(a)
                    #mol.AddAtom(Chem.Atom(atom))

                # add all bonds
                for bond in bonds:
                    #mol.AddBond(bond[0], bond[1], bond[2])
                    mol.AddBond(atom_list.index(str(bond[0])), atom_list.index(str(bond[1])), bond[2])

                # generate list of LAMMPS-IDs of the constituent atoms
                mol_frags_ids = Chem.GetMolFrags(mol,sanitizeFrags=False, asMols=False)
                mol_lmp_ids = list(mol_frags_ids).copy()

                for i, frg in enumerate(mol_frags_ids):
                    lmp_ids = []
                    for j in frg:
                        lmp_ids.append(atom_list[j])
                    mol_lmp_ids[i] = lmp_ids

                # generate list of molecules in SMILES notation
                mol_frags = Chem.GetMolFrags(mol,sanitizeFrags=False, asMols=True)
                # smiles_list = [Chem.MolToSmiles(frag) for frag in mol_frags] # creates implicit H
                smiles_list = []
                for frag in mol_frags:
                    frag = Chem.AddHs(frag)  # make every H explicit, otherwise rdkit add implicit H
                    smiles_list.append(Chem.MolToSmiles(frag, allHsExplicit=True))
                chem_formula_list = [Descriptors.rdMolDescriptors.CalcMolFormula(frag) for frag in mol_frags]

                # Update output dictionaries:
                smiles[timestep] = smiles_list
                smiles_atoms[timestep] = mol_lmp_ids
                chem_formulas[timestep] = chem_formula_list

                for i, frg in enumerate(mol_lmp_ids):
                    for j in frg:
                        atom_evolution[j].append(smiles_list[i])

                # reset key variables for new timestep
                atoms = {}
                bonds = []
                counter = 0
                mol = Chem.RWMol()

    return atom_evolution, smiles, smiles_atoms, chem_formulas

def first_appearance(dict):
    """
    function to obtain the unique list of smiles or chemical formulas as well as
    the time and ID at which they happen
    Parameters
    ----------
    dict: dictionary
        dictionary containing the respective molecular composition at each timestep
        keys:timesteps - values: [mol_t1, mol2,...,mol_n]

    Returns
    -------
    unique: list
        list of unique smiles or chemical formulas present during the simulation
    first_appearance: dict
        refers to the timestep and index a molecule appears in the original dict
        keys: SMILE/chemical formula - value: [time,mol_ID]
    """

    unique =set(chain.from_iterable(dict.values()))
    remains = unique.copy()
    first_appearance = {}

    for time, molecules in dict.items():
        if not remains:
            break # exit search when all molecules have been found

        for id, mol in enumerate(molecules):
            if mol in remains:
                first_appearance[mol] = [time,id]
                remains.discard(mol)

    return sorted(unique), first_appearance


def parse_traj(filename):
    """
    A function generator that yields one frame at a time
    to prevent loading large files into RAM. The frame contains the
    charge and the wrapped xyz-coordinates of each atom for each timestep

    Since this is a generator function to obtain the 3D array of timesteps x atom x parameters
    the function should be called like: result = np.array([frame for frame in parse_traj(traj_file)])
    Parameters
    ----------
    filename: str
        path/name of the trajectory file

    Returns
    -------
    wrapped: np.array
        frame of system at a single timestep containing charge and wrapped xyz-coordinates

    """
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: break

            # Register number of atoms in frame
            if "NUMBER" in line:
                n_atoms = int(f.readline())

            # Parse Box Bounds to get boxlengths L and min_coords
            if "ITEM: BOX BOUNDS" in line:
                bounds = []
                for _ in range(3):
                    bounds.append([float(x) for x in f.readline().split()])
                bounds = np.array(bounds)

                # min_coords = [x_lo, y_lo, z_lo]
                min_coords = bounds[:, 0]
                # box_lengths = [Lx, Ly, Lz]
                box_lengths = bounds[:, 1] - bounds[:, 0]

            # Parse Atom Data and get unwrapped coordinates
            if "ITEM: ATOMS" in line:
                cols = line.split()[2:]

                frame_data = []
                for _ in range(n_atoms):
                    atom_line = f.readline().split()
                    frame_data.append([float(atom_line[cols.index("q")]),
                                       float(atom_line[cols.index("xu")]),
                                       float(atom_line[cols.index("yu")]),
                                       float(atom_line[cols.index("zu")])])
                    # frame_data.append([float(x) for x in atom_line[3:6]])

                unwrapped = np.array(frame_data)
                wrapped = unwrapped.copy()

                # Calculate wrapped coordinates: x_w = x_min + (x_u - x_min) % L
                wrapped[:, 1:] = min_coords + (unwrapped[:, 1:] - min_coords) % box_lengths

                yield  wrapped


def get_boxdim(traj_file,target_time):
    """
    function to obtain the boxdimensions at a specified target time
    Parameters
    ----------
    traj_file: str
        name/path of the trajectory file
    target_time: int
        timestep at which box dimensions are desired

    Returns
    -------
    L: np.array
        array with the boy sidelengths at target_time
    """


    with open(traj_file) as f:
        while True:
            line = f.readline()
            if not line:
                print(f"Timesteps doesnt exist!")
                break  # Exit loop once file ends

            if "TIMESTEP" in line:
                current_time = int(f.readline().strip())

                if current_time == target_time:
                    L = []

                    while True:
                        line = f.readline()
                        if line.startswith("ITEM: BOX BOUNDS"):
                            # The next 3 lines are X, Y, and Z bounds
                            x_b = f.readline().split()
                            y_b = f.readline().split()
                            z_b = f.readline().split()
                            L.append(float(x_b[1]) - float(x_b[0]))
                            L.append(float(y_b[1]) - float(y_b[0]))
                            L.append(float(z_b[1]) - float(z_b[0]))

                            return np.array(L)


def min_img_distance(coords,traj_file,target_time=0):
    """
    function to calculate the minimum image distance of the molecular system
    Parameters
    ----------
    coords: np.array
        array containing the 3D-coordinates of the molecules
    traj_file: str
        name/path of the trajectory file
    target_time: int
        timestep of the current trajectory to detect box dimensions

    Returns
    -------
    r: np.array
        n_atom x n_atom np.array with pairwise distances
    L: np.array
        array with the boy sidelengths at target_time
    """

    # calculate pairwise displacement vector for each atom
    disp = coords[np.newaxis,:,:] - coords[:,np.newaxis,:]

    # get box lengths at target timestep
    L = get_boxdim(traj_file,target_time)
    marker = np.round(disp/L)
    disp -= L * marker

    r = np.linalg.norm(disp,axis=2)

    return r, L


def extract_unique_reaction_paths(atom_evolution):
    """
    function to extract unique reaction pathways during the simulation
    Parameters
    ----------
    atom_evolution: dictionary
        dictionary that show to which molecule each element belongs at each timestep
        -> Keys: ElementID - Values:[mol_t1,mol2,...,mol_n]

    Returns
    -------
    reactions: dictionary
        dictionary of products for each reactant, both in SMILES notation
        -> Keys: reactant - Values: [product1,product2,...,productN]
    """

    reactions = {}
    for molecules in atom_evolution.values():
        for i in range(len(molecules)-1):
            reactant = molecules[i]
            product = molecules[i+1]
            if reactant not in reactions:
                reactions[reactant] = set()
            if reactant != product:
                reactions[reactant].add(product)

    return reactions

def get_spans(indexes):
    """
    function to calculate the spans of different index groups - creates
    a list of two-member lists containing the start and end ID of the group
    Parameters
    ----------
    indexes: list
        list of indexes

    Returns
    -------
    span: list
        list of index pairs ((start_id1,end_id1),...,(start_idn,end_idn))
    """
    if not indexes:
        return []

    spans = []
    start = indexes[0]
    end = indexes[0]

    for i in indexes[1:]:
        if i == end + 1:
            end = i  # extend current span
        else:
            spans.append((start, end))  # gap found, save span
            start = i  # start new span
            end = i

    spans.append((start, end))  # adding last span
    return spans

def get_elem_ids(traj_file, type_id):
    """

    Parameters
    ----------
    traj_file: str
        name/path of the trajectory file
    type_id: int
        ID of the atom type specified in the initial LAMMPS setup

    Returns
    -------
    elem_ids: list
        list of the ElementID of the system
    """

    elem_ids = []
    with open(traj_file) as f:
        while True:
            line = f.readline()

            # Register number of atoms in frame
            if "NUMBER" in line:
                n_atoms = int(f.readline())
            # Generate list of indexes for target element
            if "ITEM: ATOMS" in line:
                for _ in range(n_atoms):
                    parts = f.readline().split()
                    if int(parts[1]) == type_id:
                        elem_ids.append(int(parts[0]))

                return elem_ids


def g_r(distances,elem_a_id, elem_b_id, boxdim, binwidth = None, r_max = None):
    """
    function to calculate the (partial) radial distribution function
    Parameters
    ----------
    distances: np.array
        array containing the pairwise distances of the whole system
    elem_a_id: list
        list of the Element ID for all type "a" elements
    elem_b_id: list
        list of the Element ID for all type "b" elements
    boxdim: list/1D np.array
        list containing the box sidelengths of the system
    binwidth: float
        binwidth of the histogram
    r_max: float
        maximum radial distance considered

    Returns
    -------
    bin_centers: np.array
        bin centers of the histogram for plotting
    rdf: np.array
        values of the normalized radial distribution for each bin center
    """

    n_a = len(elem_a_id)
    n_b = len(elem_b_id)

    dist_slice = distances[elem_a_id][:,elem_b_id]
    # calculate box volume
    V = boxdim[0] * boxdim[1] * boxdim[2]

    # prepare flattened distance array and normalization density
    if elem_a_id == elem_b_id:
        flattened_dist = dist_slice[np.triu_indices(n_a,k=1)]
        rho_bulk = (n_a -1) * 0.5/V

    else:
        flattened_dist = dist_slice.flatten()
        rho_bulk = n_b/V

    # Apply Scott's rule for optimal binwidth
    if binwidth is None:
        std = np.std(flattened_dist)
        binwidth = 3.5 * std / (len(flattened_dist)**(1/3))

    # Generate Histogram
    if r_max is None:
        r_max = min(boxdim) / 2
    bins = np.arange(0, r_max+binwidth, binwidth)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(flattened_dist, bins=bins)

    # Calculate the distinct shell volumes
    shells = 4 * np.pi * bin_centers**2 * binwidth

    # calculate local density of elements A around elements B
    rho_ab = counts / (n_a * shells)

    rdf = rho_ab / rho_bulk

    return bin_centers, rdf


class UnionFindReax():
    """

    """

    def __init__(self):
        self.root = {}

    def find(self, val):
        if val not in self.root:
            self.root[val] = val
        if self.root[val] != val:
            self.root[val] = self.find(self.root[val])
        return self.root[val]

    def union(self,val1,val2):

        root_val1, root_val2 = self.find(val1), self.find(val2)

        if root_val1 != root_val2:
            self.root[root_val1] = root_val2

def reaction_clusters(uf_object, mol_list_t1,mol_list_t2):

    for i,mol in enumerate(mol_list_t1):
        for j in mol:
            uf_object.union(('reactant', i), ('product', j))

    reactions = defaultdict(lambda: {'reactants': [], 'products': []})

    for i in range(len(mol_list_t1)):
        root_mol = uf_object.find(("reactant", i))
        reactions[root_mol]['reactants'].append(i)

    for j in range(len(mol_list_t2)):
        root_mol = uf_object.find(("product", j))
        reactions[root_mol]['products'].append(j)

    return list(reactions.values())

def map_atoms_to_mols(smiles_list, ids_list):
    atom_to_mol = {}
    for idx, ids in enumerate(ids_list):
        for atom in ids:
            atom_to_mol[atom] = (smiles_list[idx], idx)
    return atom_to_mol

# def angle