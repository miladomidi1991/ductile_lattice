# -*- coding: utf-8 -*-
"""
Kagome lattice builder with PRE-MESH cell-wall removal
- Builds a 2D Kagome lattice (instanced & merged into a single wire Part)
- BEFORE meshing: removes a random fraction of wire edges whose length ≈ L
- Proceeds with sectioning, meshing, orphan conversion, BCs, and job write

Note: Python 2.7-compatible (no f-strings)
"""

from abaqus import *
from abaqusConstants import *
import job, mesh, part, regionToolset, sketch
import math, random, time

t0 = time.time()

# === GLOBAL PARAMETERS ===
L = 1.00                       # Bar length [mm]
elem_N = 5
el_size = L/elem_N             # Mesh element size [mm]
Bi = 1.0                       # Beam width 'a' [mm]
p = 0.05                       # Relative density

Vy = 10.0                      # Loading velocity [mm/s]

Nx = 8                        # Hexagons in X
Ny = int(round(11.0 / math.sqrt(3.0) * Nx)/2.00)  # Hexagons in Y

failure_strain = 0.0015
Period = failure_strain * int(round(11.0 / math.sqrt(3.0) * Nx)/2.00) * math.sqrt(3.0) * L / Vy

max_r = 0.1 * L                # Pre-geometry perturbation (not used for wire stage here)
material_type = 'Brittle'      # 'Brittle' or 'Ductile'

# Removal settings (applied to walls at the GEOMETRY stage)
removal_fraction = 0.05        # Fraction of wire edges (length ≈ L) to remove
p = p * (1+removal_fraction)    # Adjusted density for removal
t = p * L / math.sqrt(3.0)     # Beam thickness 'b' [mm]
wall_len_tol = 0.15            # Relative tolerance on length to classify walls

# Multi-section assignment controls
N_SECTIONS = 20                # Number of distinct thickness sections (adjustable)
thickness_var_pct = 0.10       # A in (t ± A*t)

# Derived spacing for flat-topped hex tiling
dx = 2.0 * L
dy = math.sqrt(3.0) * L
dy_vertex = math.sqrt(3.0)/2.0 * L

# Flat-topped hexagon offsets from center
hex_offsets = [
    ( L, 0.0),
    ( 0.5*L,  dy_vertex),
    (-0.5*L,  dy_vertex),
    (-L, 0.0),
    (-0.5*L, -dy_vertex),
    ( 0.5*L, -dy_vertex),
]

# === MATERIAL PROPS ===
ps = 7.8e-9                    # Density [g/mm^3]
Es = 200000.0                  # Young's modulus [MPa]
S_0s = 200.0                   # Yield [MPa]
vs = 0.3
Ef = 1e-6                      # Failure strain used for DD initiation
Uf = Ef * el_size              # Failure displacement [mm]
nbr_frames = 200               # Field output frames


# ============================================================
# ===============   GEOMETRY (INSTANCED MERGE)   =============
# ============================================================
model_name = 'Kagome%dX_%dY' % (Nx, Ny)
Model = mdb.Model(name=model_name)

# Build a reusable unit hexagon part (wire)
unit_part_name = 'UnitHex'
UnitPart = Model.Part(name=unit_part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
hex_pts3d = []
for off in hex_offsets:
    hex_pts3d.append((off[0], off[1], 0.0))
unit_edges = []
for k in range(6):
    p1 = hex_pts3d[k]
    p2 = hex_pts3d[(k + 1) % 6]
    unit_edges.append((p1, p2))
UnitPart.WirePolyLine(points=tuple(unit_edges), mergeType=IMPRINT, meshable=ON)

# Tile instances with staggered rows
asm = Model.rootAssembly
instances_by_row = {}
for jj in range(Ny):
    y = jj * dy
    x_offset = 0 if jj % 2 == 0 else L
    row_instances = []
    for i in range(Nx):
        x = i * dx + x_offset
        inst_name = 'inst_%d_%d' % (i, jj)
        inst = asm.Instance(name=inst_name, part=UnitPart, dependent=ON)
        inst.translate(vector=(x, y, 0.0))
        row_instances.append(inst)
    instances_by_row[jj] = row_instances

# Merge each row to reduce boolean complexity
row_part_names = []
for jj in range(Ny):
    row_insts = instances_by_row[jj]
    if not row_insts:
        continue
    row_name = 'RowMerge_%d' % jj
    asm.InstanceFromBooleanMerge(
        name=row_name,
        instances=tuple(row_insts),
        originalInstances=DELETE,
        mergeNodes=BOUNDARY_ONLY,
        nodeMergingTolerance=1.0e-6,
        domain=BOTH
    )
    row_part_names.append(row_name)

# Final merge across rows
final_instances = []
for row_name in row_part_names:
    row_part = Model.parts[row_name]
    row_inst = asm.Instance(name='inst_' + row_name, part=row_part, dependent=ON)
    final_instances.append(row_inst)

merged_name = 'MergedKagome'
asm.InstanceFromBooleanMerge(
    name=merged_name,
    instances=tuple(final_instances),
    originalInstances=DELETE,
    mergeNodes=BOUNDARY_ONLY,
    nodeMergingTolerance=1.0e-6,
    domain=BOTH
)

# Use the merged lattice as the active Part (wire geometry)
PartMerged = Model.parts[merged_name]


# ============================================================
# ========   PRE-MESH WALL REMOVAL AT GEOMETRY LEVEL   =======
# ============================================================
def _vertex_point(v):
    try:
        # Vertex object
        return v.pointOn[0]
    except Exception:
        # Integer index into part vertices
        vv = PartMerged.vertices[v]
        try:
            return vv.pointOn[0]
        except Exception:
            return vv.coordinates

def edge_length(edge_obj):
    try:
        # Most robust for CAE Part Edge
        return edge_obj.getSize(printResults=False)
    except Exception:
        # Fallback via vertex coordinates
        vtx = edge_obj.getVertices()
        if len(vtx) != 2:
            return 0.0
        p1 = _vertex_point(vtx[0])
        p2 = _vertex_point(vtx[1])
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

# Classify wall edges by length ~ L (use integer indices instead of Edge objects)
candidate_edge_idx = []
for _idx, _e in enumerate(PartMerged.edges):
    le = edge_length(_e)
    if abs(le - L) <= wall_len_tol * L:
        candidate_edge_idx.append(_idx)

total_walls = len(candidate_edge_idx)
num_remove = int(total_walls * removal_fraction)
edges_to_remove_idx = set(random.sample(candidate_edge_idx, num_remove)) if num_remove > 0 else set()
print('Pre-mesh walls: %d; scheduled removals: %d' % (total_walls, len(edges_to_remove_idx)))

# Pre-mesh nodal dislocation at junctions (degree >= 3) on wire geometry
pre_dislocated_count = 0
disloc_map = {}
def _vkey(pt):
    return (round(pt[0], 6), round(pt[1], 6))

if max_r > 0.0:
    # Build vertex degree from PartMerged edges
    vdeg = {}
    for e in PartMerged.edges:
        vtx = e.getVertices()
        if len(vtx) != 2:
            continue
        p1 = _vertex_point(vtx[0])
        p2 = _vertex_point(vtx[1])
        k1 = _vkey(p1)
        k2 = _vkey(p2)
        vdeg[k1] = vdeg.get(k1, 0) + 1
        vdeg[k2] = vdeg.get(k2, 0) + 1
    # Create random radial offsets for junctions only
    def _rand_rad(dx_max):
        th = random.uniform(0.0, 2.0*math.pi)
        r  = random.uniform(0.0, dx_max)
        return (r*math.cos(th), r*math.sin(th))
    for k, d in vdeg.items():
        if d >= 3:
            rx, ry = _rand_rad(max_r)
            disloc_map[k] = (k[0] + rx, k[1] + ry)
    pre_dislocated_count = len(disloc_map)
    if pre_dislocated_count > 0:
        print('Pre-mesh junction dislocation applied at %d vertices (max_r=%.4f).' % (pre_dislocated_count, max_r))

# Rebuild a filtered wire Part from kept edges (applying pre-mesh dislocation to endpoints)
kept_segments = []
for _idx, e in enumerate(PartMerged.edges):
    if _idx in edges_to_remove_idx:
        continue
    vtx = e.getVertices()
    if len(vtx) != 2:
        continue
    p1 = _vertex_point(vtx[0])
    p2 = _vertex_point(vtx[1])
    k1 = _vkey(p1)
    k2 = _vkey(p2)
    q1 = disloc_map.get(k1, (p1[0], p1[1]))
    q2 = disloc_map.get(k2, (p2[0], p2[1]))
    kept_segments.append(((q1[0], q1[1], 0.0), (q2[0], q2[1], 0.0)))

FilteredPart = Model.Part(name='KagomeWireFiltered', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
if kept_segments:
    FilteredPart.WirePolyLine(points=tuple(kept_segments), mergeType=IMPRINT, meshable=ON)
print('Pre-mesh walls removed: %d; kept segments: %d' % (len(edges_to_remove_idx), len(kept_segments)))

# Optionally drop the merged source Part to keep only filtered
try:
    del Model.parts[merged_name]
except Exception:
    pass

# Adopt filtered Part for downstream steps
Part = FilteredPart


# ============================================================
# ===============   SECTIONS / MESH (NATIVE)   ===============
# ============================================================
mat = Model.Material(name='damage')
mat.Density(table=((ps,),))
mat.Elastic(table=((Es, vs),))
if material_type == 'Ductile':
    mat.DuctileDamageInitiation(table=((Ef, 0.0, 0.0),))
    mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((Uf,),), viscosity=1.0e-6)
    mat.Plastic(table=((S_0s, 0.0),))
elif material_type == 'Brittle':
    mat.DuctileDamageInitiation(table=((Ef, 0.0, 0.0),))
    mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((Uf,),))
    mat.Plastic(table=((S_0s, 0.0),))
else:
    print('Unknown material_type')

# Identify wall edges on filtered Part (length ≈ L)
wall_edge_idx = []
for ei, e in enumerate(Part.edges):
    try:
        le = e.getSize(printResults=False)
    except Exception:
        vtx = e.getVertices()
        if len(vtx) != 2:
            continue
        p1 = _vertex_point(vtx[0])
        p2 = _vertex_point(vtx[1])
        dxl = p2[0]-p1[0]; dyl = p2[1]-p1[1]; dzl = p2[2]-p1[2]
        le = math.sqrt(dxl*dxl + dyl*dyl + dzl*dzl)
    if abs(le - L) <= wall_len_tol * L:
        wall_edge_idx.append(ei)

# Build N section thicknesses within t ± A*t (evenly spaced)
th_min = t * (1.0 - thickness_var_pct)
th_max = t * (1.0 + thickness_var_pct)
if N_SECTIONS <= 1:
    thicknesses = [t]
else:
    thicknesses = [th_min + i*(th_max - th_min)/(N_SECTIONS - 1) for i in range(N_SECTIONS)]

random.shuffle(wall_edge_idx)
X_total = len(wall_edge_idx)
base = X_total // max(1, N_SECTIONS)
rem = X_total % max(1, N_SECTIONS)
assignments = []
cursor = 0
for i in range(N_SECTIONS):
    take = base + (1 if i < rem else 0)
    block = wall_edge_idx[cursor:cursor+take]
    assignments.append(block)
    cursor += take

print('Section assignment: total walls=%d, classes=%d, base t=%.5f, range=[%.5f, %.5f]' % (len(wall_edge_idx), N_SECTIONS, t, th_min, th_max))

assigned_any = False
assigned_counts = {}
assigned_indices = set()
created_sets = []
# Create profiles/sections and assign per class
for i, edge_ids in enumerate(assignments):
    if not edge_ids:
        continue
    prof_name = 'Rect_%d' % i
    sec_name  = 'BeamSection_%d' % i
    bi = Bi
    bj = thicknesses[min(i, len(thicknesses)-1)]
    try:
        Model.RectangularProfile(name=prof_name, a=bi, b=bj)
    except Exception:
        pass
    try:
        Model.BeamSection(name=sec_name, integration=DURING_ANALYSIS, profile=prof_name, material='damage')
    except Exception:
        pass
    # Build unique, valid edge references
    uniq_ids = []
    seen = set()
    for k in edge_ids:
        if isinstance(k, (int, long)) and 0 <= k < len(Part.edges) and k not in seen:
            uniq_ids.append(k); seen.add(k)
    if not uniq_ids:
        continue
    reg_edges = tuple(Part.edges[k] for k in uniq_ids)
    set_name = 'WallSet_%d' % i
    try:
        if set_name in Part.sets.keys():
            del Part.sets[set_name]
    except Exception:
        pass
    try:
        # Create set first (CAE-style), then assign section using that set
        reg = Part.Set(name=set_name, edges=reg_edges)
        region_edges = regionToolset.Region(edges=reg_edges)
        Part.SectionAssignment(region=region_edges, sectionName=sec_name,
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        assigned_any = True
        assigned_counts[i] = len(reg_edges)
        created_sets.append(set_name)
        for k in uniq_ids:
            assigned_indices.add(k)
    except Exception:
        # Fallback: assign in smaller chunks; if still failing, assign per-edge
        chunk = 500
        success = False
        try:
            for k0 in range(0, len(uniq_ids), chunk):
                sub_ids = uniq_ids[k0:k0+chunk]
                if not sub_ids:
                    continue
                sub_edges = tuple(Part.edges[k] for k in sub_ids)
                sub_name = '%s_%d' % (set_name, k0//chunk)
                try:
                    if sub_name in Part.sets.keys():
                        del Part.sets[sub_name]
                    Part.Set(name=sub_name, edges=sub_edges)
                    region_sub = regionToolset.Region(edges=sub_edges)
                    Part.SectionAssignment(region=region_sub, sectionName=sec_name,
                        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
                    created_sets.append(sub_name)
                except Exception:
                    pass
            assigned_any = True
            assigned_counts[i] = len(reg_edges)
            success = True
            for k in sub_ids:
                assigned_indices.add(k)
        except Exception:
            success = False
        if not success:
            # Final fallback: per-edge assignment
            for eid in uniq_ids:
                try:
                    e = Part.edges[eid]
                    one_name = '%s_e%d' % (set_name, eid)
                    try:
                        if one_name in Part.sets.keys():
                            del Part.sets[one_name]
                        Part.Set(name=one_name, edges=(e,))
                        region_one = regionToolset.Region(edges=(e,))
                        Part.SectionAssignment(region=region_one, sectionName=sec_name,
                            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
                        created_sets.append(one_name)
                    except Exception:
                        pass
                    assigned_any = True
                    assigned_counts[i] = assigned_counts.get(i, 0) + 1
                    assigned_indices.add(eid)
                except Exception:
                    # Skip problematic edge
                    pass

# Assign beam orientation to all edges
regAll = Part.Set(name='EntireLatticeSet', edges=Part.edges)
Part.assignBeamSectionOrientation(region=regAll, method=N1_COSINES, n1=(0.0,0.0,1.0))

# Print per-class counts and thickness
for i, edge_ids in enumerate(assignments):
    th_i = thicknesses[min(i, len(thicknesses)-1)]
    assigned_edges = assigned_counts.get(i, 0)
    print('Class %d: walls=%d, thickness=%.5f, assigned_edges=%d' % (i, len(edge_ids), th_i, assigned_edges))

print('Total section assignments on Part before mesh: %d' % len(Part.sectionAssignments))
print('Created sets (count=%d): %s' % (len(created_sets), ', '.join(created_sets)))

# Ensure complete coverage: assign any remaining wall edges to the last section
remaining = [k for k in wall_edge_idx if k not in assigned_indices]
if remaining:
    last_i = min(N_SECTIONS - 1, len(thicknesses) - 1)
    last_sec = 'BeamSection_%d' % last_i
    rem_edges = tuple(Part.edges[k] for k in remaining)
    try:
        region_rem = regionToolset.Region(edges=rem_edges)
        Part.SectionAssignment(
            region=region_rem,
            sectionName=last_sec,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField='',
            thicknessAssignment=FROM_SECTION
        )
        try:
            if 'WallSet_remaining' in Part.sets.keys():
                del Part.sets['WallSet_remaining']
            Part.Set(name='WallSet_remaining', edges=rem_edges)
            created_sets.append('WallSet_remaining')
        except Exception:
            pass
        print('Assigned remaining %d wall edges to %s' % (len(remaining), last_sec))
    except Exception:
        # Fallback per-edge
        for eid in remaining:
            try:
                e = Part.edges[eid]
                Part.SectionAssignment(
                    region=regionToolset.Region(edges=(e,)),
                    sectionName=last_sec,
                    offset=0.0,
                    offsetType=MIDDLE_SURFACE,
                    offsetField='',
                    thicknessAssignment=FROM_SECTION
                )
            except Exception:
                pass

Part.seedPart(size=el_size)
# Ensure compatibility with downstream variable name
regN = Part.sets['EntireLatticeSet']
Part.setElementType(regions=regN, elemTypes=(mesh.ElemType(elemCode=B21),))
Part.generateMesh()


# ============================================================
# ======  ORPHAN CONVERSION (NO post-mesh deletions)  ========
# ============================================================
asm = Model.rootAssembly
instNat = asm.Instance(name='Inst_native', part=Part, dependent=ON)
orphan_name = 'OrphanKagome'
orphanPart = asm.PartFromInstanceMesh(name=orphan_name, partInstances=(instNat,))

# Apply nodal dislocation only at junction nodes (degree >= 3)
def _random_radial(dx_max):
    if dx_max <= 0.0:
        return (0.0, 0.0)
    th = random.uniform(0.0, 2.0*math.pi)
    r  = random.uniform(0.0, dx_max)
    return (r*math.cos(th), r*math.sin(th))

if max_r > 0.0 and pre_dislocated_count == 0:
    # Build node degree from element connectivity
    node_degree = {}
    for el in orphanPart.elements:
        conn = el.connectivity
        if len(conn) >= 2:
            a, b = conn[0], conn[1]
            node_degree[a] = node_degree.get(a, 0) + 1
            node_degree[b] = node_degree.get(b, 0) + 1

    # Select only nodes that act as junctions (degree >= 3)
    node_batch = []
    coord_batch = []
    moved_count = 0
    for nd in orphanPart.nodes:
        lbl = nd.label
        deg = node_degree.get(lbl, 0)
        if deg >= 3:
            x, y, z = nd.coordinates
            rx, ry = _random_radial(max_r)
            node_batch.append(nd)
            coord_batch.append((x + rx, y + ry, z))
            moved_count += 1
    if moved_count > 0:
        chunk = 5000
        for k0 in range(0, len(node_batch), chunk):
            orphanPart.editNode(nodes=tuple(node_batch[k0:k0+chunk]), coordinates=tuple(coord_batch[k0:k0+chunk]))
    print('Applied nodal dislocation to %d junction nodes (max_r=%.4f).' % (moved_count, max_r))
elif max_r > 0.0 and pre_dislocated_count > 0:
    print('Skipped orphan-node dislocation (already applied pre-mesh at %d junctions).' % pre_dislocated_count)

# Reassign section/orientation to remaining elements
elemRegion = regionToolset.Region(elements=orphanPart.elements)
try:
    _ = Model.sections['BeamSection']
    orphanPart.SectionAssignment(region=elemRegion, sectionName='BeamSection')
    print('Assigned default BeamSection to orphan elements.')
except Exception:
    print('No default BeamSection found; keeping pre-mesh per-class sections on orphan.')
orphanPart.assignBeamSectionOrientation(region=elemRegion, method=N1_COSINES, n1=(0.0,0.0,1.0))

# Swap in orphan instance and cleanup (suppress old before creating new)
asm.features.changeKey(fromName='Inst_native', toName='Inst_old')
try:
    asm.features['Inst_old'].suppress()
except Exception:
    try:
        del asm.features['Inst_old']
    except Exception:
        pass
try:
    asm.regenerate()
except Exception:
    pass
asm.Instance(name='Inst', part=orphanPart, dependent=ON)
inst = asm.instances['Inst']

# Remove intermediate parts so only orphan remains
try:
    row_names = [n for n in Model.parts.keys() if n.startswith('RowMerge_')]
    for rn in row_names:
        try:
            inst_keys = [ik for ik, iv in asm.instances.items() if iv.partName == rn]
            for ik in inst_keys:
                asm.features[ik].suppress()
                del asm.features[ik]
        except Exception:
            pass
        del Model.parts[rn]
except Exception:
    pass
try:
    if unit_part_name in Model.parts.keys():
        del Model.parts[unit_part_name]
except Exception:
    pass
try:
    if 'Inst_old' in asm.features.keys():
        del asm.features['Inst_old']
except Exception:
    pass


# ============================================================
# ===================   STEP & OUTPUTS   =====================
# ============================================================
step_name = 'damage'
Model.ExplicitDynamicsStep(name=step_name, previous='Initial', timePeriod=Period, improvedDtMethod=ON)

Model.FieldOutputRequest(name='F-Output-1', createStepName=step_name, variables=('RF','U'))
Model.fieldOutputRequests['F-Output-1'].setValues(numIntervals=nbr_frames)
Model.FieldOutputRequest(name='F-Output-2', createStepName=step_name, variables=('LE','S','SDEG','STATUS'))
Model.fieldOutputRequests['F-Output-2'].setValues(numIntervals=nbr_frames)


# ============================================================
# ==================   RP, TIE, & BCs   ======================
# ============================================================
W_x = 2.0 * Nx * L
W_y = Ny * math.sqrt(3.0) * L

rp = asm.ReferencePoint(point=((W_x - L)/2.0, W_y - math.sqrt(3.0)*L/2.0, 0.0))
Ref1 = asm.referencePoints[rp.id]
asm.Set(referencePoints=(Ref1,), name='Ref1')
regionRP = asm.sets['Ref1']

tie_nodes = inst.nodes.getByBoundingBox(
    xMin=-L/2.0, yMin=W_y - math.sqrt(3.0)*L/2.0 - el_size/10.0, zMin=-1.0,
    xMax=W_x,     yMax=W_y + el_size/10.0,                          zMax= 1.0
)
tieRegion = regionToolset.Region(nodes=tie_nodes)
Model.RigidBody(name='Constraint-1', refPointRegion=regionRP, tieRegion=tieRegion)

Model.VelocityBC(name='Up', createStepName=step_name, region=regionRP, v2=Vy)

bottom_nodes = inst.nodes.getByBoundingBox(
    xMin=-L/2.0 - el_size,   yMin=-L*math.sqrt(3.0)/2.0 - el_size, zMin=-1.0,
    xMax=W_x + el_size,      yMax=-L*math.sqrt(3.0)/2.0 + el_size, zMax= 1.0
)
botRegion = regionToolset.Region(nodes=bottom_nodes)
Model.DisplacementBC(name='Bottom', createStepName=step_name, region=botRegion, u2=0.0)

LR_nodes = inst.nodes.getByBoundingBox(
    xMin=-L, yMin=-L*math.sqrt(3.0)/2.0, zMin=-1.0,
    xMax=-L, yMax=W_y - math.sqrt(3.0)*L/2.0, zMax= 1.0
)
LRRegion = regionToolset.Region(nodes=LR_nodes)
Model.DisplacementBC(name='Left-and-right', createStepName=step_name, region=LRRegion, u1=0.0)


# ============================================================
# ===================   WRITE INPUT   ========================
# ============================================================
print('Writing job input...')
print('Lattice size: W = %.3f, H = %.3f' % (W_x, W_y))
ru_token   = ('%.4f' % p).replace('.', 'p')
del_token  = ('%.4f' % removal_fraction).replace('.', 'p')
dis_token  = ('%.4f' % (max_r / L)).replace('.', 'p')
sec_token  = '%d' % int(N_SECTIONS)
var_token  = ('%.3f' % thickness_var_pct).replace('.', 'p')
job_name = 'UNI_Kagome_pre_sec%s_var%s_ru%s_dis%s_del%s_Nx%d' % (sec_token, var_token, ru_token, dis_token, del_token, Nx)
mdb.Job(name=job_name, model=model_name, numCpus=6, numDomains=6).writeInput()
print('Job name: %s' % job_name)

print("Elapsed time: %.2f seconds" % (time.time() - t0))


