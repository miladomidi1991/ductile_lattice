# -*- coding: utf-8 -*-
"""
OPTIMIZED Kagome parametric study - minimal features for speed
Generates models with names using 'p' for decimals.
"""

from abaqus import *
from abaqusConstants import *
from ka_imp_prewall import S_0s
import job, mesh, part, regionToolset
import math, random, time

# ===== OPTIMIZED PARAMETERS =====
L = 1.00
elem_N = 5
el_size = L/elem_N
Bi = 1.0
Vy = 15.0
num_intervals = 500   # Field output frames    

Nx = 40
Ny = int(round(11.0 / math.sqrt(3.0) * Nx)/2.00)

failure_strain = 0.02
Period_base = failure_strain * int(round(11.0 / math.sqrt(3.0) * Nx)/2.00) * math.sqrt(3.0) * L / Vy

# Simplified material (single values)
ps = 7.8e-9       # Density
Es = 200000.0      # Young's modulus
S_0s = 200.0       # Yield strength
vs = 0.3           # Poisson's ratio
Ef = 1e-6          # Failure strain
wall_len_tol = 0.15

# ===== PARAMETER GRIDS =====
ru_list = [0.01]
removal_fraction_list = [0.00, 0.01, 0.05]
max_r_mult_list = [0.0, 0.1, 0.3]
thickness_var_pct_list = [0.0, 0.1, 0.3]
N_SECTIONS = 15  # Reduced from 10 for speed
DISLOC_FRACTION = 1.0  # Restored for parametric study
BASE_SEED = 12345  # Deterministic seed for invariant spatial patterns


def token(x):
    return ('%.4f' % x).rstrip('0').rstrip('.').replace('.', 'p')


def build_one(ru_base, removal_fraction, max_r_mult, thickness_var_pct, idx):
    t0 = time.time()
    print('Building model %d...' % idx)
    
    # Derived
    max_r = max_r_mult * L
    ru = ru_base * (1.0 + removal_fraction)
    t = ru * L / math.sqrt(3.0)
    Period = Period_base

    # Model / job names without dots
    ru_tok = token(ru_base)
    del_tok = token(removal_fraction)
    dis_tok = token(max_r_mult)
    var_tok = token(thickness_var_pct)
    model_name = 'Kagome_%02d_ru%s_del%s_dis%s_var%s_Nx%d' % (idx, ru_tok, del_tok, dis_tok, var_tok, Nx)
    job_name = 'UNI_Kagome_ru%s_del%s_dis%s_var%s_Nx%d' % (ru_tok, del_tok, dis_tok, var_tok, Nx)

    if model_name in mdb.models.keys():
        del mdb.models[model_name]
    Model = mdb.Model(name=model_name)
    
    t1 = time.time()
    print('  Model setup: %.2fs' % (t1-t0))

    # OPTIMIZED: Direct geometry creation (skip complex tiling)
    dx = 2.0 * L
    dy = math.sqrt(3.0) * L
    dy_vertex = math.sqrt(3.0)/2.0 * L
    
    # Create assembly first
    asm = Model.rootAssembly
    
    # Create single merged part directly
    merged_name = 'MergedKagome'
    PartMerged = Model.Part(name=merged_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    
    # Generate all segments at once
    all_segments = []
    for jj in range(Ny):
        y = jj * dy
        x_offset = 0 if jj % 2 == 0 else L
        for i in range(Nx):
            x = i * dx + x_offset
            # Hexagon vertices (3D coordinates)
            hex_pts = [
                (x + L, y, 0.0), (x + 0.5*L, y + dy_vertex, 0.0), (x - 0.5*L, y + dy_vertex, 0.0),
                (x - L, y, 0.0), (x - 0.5*L, y - dy_vertex, 0.0), (x + 0.5*L, y - dy_vertex, 0.0)
            ]
            # Hexagon edges
            for k in range(6):
                p1 = hex_pts[k]
                p2 = hex_pts[(k + 1) % 6]
                all_segments.append((p1, p2))
    
    # Create wire in one go
    if all_segments:
        PartMerged.WirePolyLine(points=tuple(all_segments), mergeType=IMPRINT, meshable=ON)
    
    t2 = time.time()
    print('  Geometry: %.2fs' % (t2-t1))

    # Helpers
    def _vertex_point(v):
        try:
            return v.pointOn[0]
        except Exception:
            vv = PartMerged.vertices[v]
            try:
                return vv.pointOn[0]
            except Exception:
                return vv.coordinates

    def edge_length(e):
        try:
            return e.getSize(printResults=False)
        except Exception:
            vtx = e.getVertices()
            if len(vtx) != 2:
                return 0.0
            p1 = _vertex_point(vtx[0]); p2 = _vertex_point(vtx[1])
            dxl = p2[0]-p1[0]; dyl = p2[1]-p1[1]; dzl = p2[2]-p1[2]
            return math.sqrt(dxl*dxl + dyl*dyl + dzl*dzl)

    def edge_midpoint(e):
        vtx = e.getVertices()
        if len(vtx) != 2:
            return (0.0, 0.0)
        p1 = _vertex_point(vtx[0]); p2 = _vertex_point(vtx[1])
        return ((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5)

    # Wall classification
    candidate_edge_idx = []
    for ei, e in enumerate(PartMerged.edges):
        le = edge_length(e)
        if abs(le - L) <= wall_len_tol * L:
            candidate_edge_idx.append(ei)

    # Spatially spread removals (deterministic across models)
    edges_to_remove_idx = set()
    total_walls = len(candidate_edge_idx)
    if total_walls > 0 and removal_fraction > 0.0:
        W_x = 2.0 * Nx * L
        N_BUCKETS = max(5, min(50, int(Nx)))
        buckets = [[] for _ in range(N_BUCKETS)]
        for ei in candidate_edge_idx:
            mx, _ = edge_midpoint(PartMerged.edges[ei])
            b = int(max(0, min(N_BUCKETS - 1, (mx / max(1e-12, W_x)) * N_BUCKETS)))
            buckets[b].append(ei)
        # Deterministic selection using per-edge seeded RNG
        for bucket in buckets:
            if not bucket:
                continue
            for ei in bucket:
                rng = random.Random(BASE_SEED + int(ei))
                if rng.random() < removal_fraction:
                    edges_to_remove_idx.add(ei)

    # Spatially spread junction dislocation (deterministic across models)
    disloc_map = {}
    if max_r > 0.0:
        vdeg = {}
        for e in PartMerged.edges:
            vtx = e.getVertices()
            if len(vtx) != 2:
                continue
            p1 = _vertex_point(vtx[0]); p2 = _vertex_point(vtx[1])
            k1 = (round(p1[0], 6), round(p1[1], 6)); k2 = (round(p2[0], 6), round(p2[1], 6))
            vdeg[k1] = vdeg.get(k1, 0) + 1
            vdeg[k2] = vdeg.get(k2, 0) + 1
        junction_pts = [k for (k, d) in vdeg.items() if d >= 3]
        for (xk, yk) in junction_pts:
            # Deterministic selection
            r_sel = random.Random(BASE_SEED ^ hash((round(xk,6), round(yk,6), 'sel'))).random()
            if r_sel <= DISLOC_FRACTION:
                # Deterministic direction and magnitude
                r_dir = random.Random(BASE_SEED ^ hash((round(xk,6), round(yk,6), 'dir'))).random()
                r_mag = random.Random(BASE_SEED ^ hash((round(xk,6), round(yk,6), 'mag'))).random()
                th = 2.0 * math.pi * r_dir
                r = max_r * r_mag
                rx = r * math.cos(th); ry = r * math.sin(th)
                disloc_map[(xk, yk)] = (xk + rx, yk + ry)

    # Rebuild filtered wire Part with dislocation
    kept_segments = []
    def _vkey(pt):
        return (round(pt[0], 6), round(pt[1], 6))
    for ei, e in enumerate(PartMerged.edges):
        if ei in edges_to_remove_idx:
            continue
        vtx = e.getVertices()
        if len(vtx) != 2:
            continue
        p1 = _vertex_point(vtx[0]); p2 = _vertex_point(vtx[1])
        k1 = _vkey(p1); k2 = _vkey(p2)
        q1 = disloc_map.get(k1, (p1[0], p1[1]))
        q2 = disloc_map.get(k2, (p2[0], p2[1]))
        kept_segments.append(((q1[0], q1[1], 0.0), (q2[0], q2[1], 0.0)))

    Part = Model.Part(name='KagomeWireFiltered', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    if kept_segments:
        Part.WirePolyLine(points=tuple(kept_segments), mergeType=IMPRINT, meshable=ON)

    # Clean up merged part (keep assembly)
    try:
        del Model.parts[merged_name]
    except Exception:
        pass
    
    t3 = time.time()
    print('  Wall removal: %.2fs' % (t3-t2))

    # Randomized, non-overlapping geometry sets Set-1..Set-N (spatially distributed, deterministic)
    p = Part
    X = len(p.edges)
    if X > 0 and N_SECTIONS > 0:
        # Clean existing sets
        for i_del in range(1, N_SECTIONS + 1):
            nm = 'Set-%d' % i_del
            if nm in p.sets.keys():
                del p.sets[nm]
        
        # Shuffle edge indices deterministically to distribute selection across the lattice
        all_idx = list(range(X))
        rng_idx = random.Random(BASE_SEED)
        rng_idx.shuffle(all_idx)
        
        # Round-robin indices into N buckets to ensure each set samples the whole domain
        buckets = [[] for _ in range(N_SECTIONS)]
        for pos, idxe in enumerate(all_idx):
            buckets[pos % N_SECTIONS].append(idxe)
        
        # For each bucket, coalesce consecutive indices into contiguous runs and create subsets
        for i in range(N_SECTIONS):
            set_name = 'Set-%d' % (i + 1)
            if set_name in p.sets.keys():
                del p.sets[set_name]
            
            idxs = sorted(buckets[i])
            if not idxs:
                p.Set(name=set_name, edges=())
                continue
            
            # Find contiguous runs
            runs = []
            run_start = idxs[0]
            prev = idxs[0]
            for k in idxs[1:]:
                if k == prev + 1:
                    prev = k
                    continue
                runs.append((run_start, prev))
                run_start = k
                prev = k
            runs.append((run_start, prev))
            
            # Create temporary subset sets for each run, then union into Set-i
            subset_names = []
            for rj, (a, b) in enumerate(runs):
                edge_slice = p.edges[a:b+1]
                sub_name = '%s_sub_%d' % (set_name, rj)
                if sub_name in p.sets.keys():
                    del p.sets[sub_name]
                try:
                    p.Set(name=sub_name, edges=edge_slice)
                    subset_names.append(sub_name)
                except Exception:
                    pass
            
            # Union subsets into the final Set-i
            if subset_names:
                try:
                    p.SetByBoolean(name=set_name, sets=tuple(p.sets[n] for n in subset_names))
                except Exception:
                    p.Set(name=set_name, edges=p.sets[subset_names[0]].edges)
            else:
                p.Set(name=set_name, edges=())
    
    t4 = time.time()
    print('  Set creation: %.2fs' % (t4-t3))

    # OPTIMIZED: Simplified material and sections
    mat = Model.Material(name='damage')
    mat.Density(table=((ps,),))
    mat.Elastic(table=((Es, vs),))
    mat.DuctileDamageInitiation(table=((Ef, 0.0, 0.0),))
    mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((Ef*el_size,),))
    mat.Plastic(table=((S_0s, 0.0),))

    # Calculate thicknesses
    th_min = t * (1.0 - thickness_var_pct)
    th_max = t * (1.0 + thickness_var_pct)
    if N_SECTIONS <= 1:
        thicknesses = [t]
    else:
        thicknesses = [th_min + i*(th_max - th_min)/(N_SECTIONS - 1) for i in range(N_SECTIONS)]

    # Assign sections to Set-1..N
    for i in range(N_SECTIONS):
        set_name = 'Set-%d' % (i + 1)
        if set_name not in Part.sets.keys():
            continue
        prof_name = 'Rect_%d' % i
        sec_name  = 'BeamSection_%d' % i
        bi = Bi
        bj = thicknesses[min(i, len(thicknesses)-1)]
        
        Model.RectangularProfile(name=prof_name, a=bi, b=bj)
        Model.BeamSection(name=sec_name, integration=DURING_ANALYSIS, profile=prof_name, material='damage')
        Part.SectionAssignment(region=Part.sets[set_name], sectionName=sec_name,
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
    
    t5 = time.time()
    print('  Sections: %.2fs' % (t5-t4))

    # OPTIMIZED: Mesh and instance
    regAll = Part.Set(name='EntireLatticeSet', edges=Part.edges)
    Part.assignBeamSectionOrientation(region=regAll, method=N1_COSINES, n1=(0.0,0.0,1.0))
    Part.seedPart(size=el_size)
    Part.setElementType(regions=regAll, elemTypes=(mesh.ElemType(elemCode=B21),))
    Part.generateMesh()
    inst = asm.Instance(name='Inst', part=Part, dependent=ON)
    
    t6 = time.time()
    print('  Mesh: %.2fs' % (t6-t5))

    # OPTIMIZED: Minimal step & outputs
    step_name = 'damage'
    Model.ExplicitDynamicsStep(name=step_name, previous='Initial', timePeriod=Period, improvedDtMethod=ON)
    Model.FieldOutputRequest(name='F-Output-1', createStepName=step_name, variables=('RF','U'))
    Model.fieldOutputRequests['F-Output-1'].setValues(numIntervals=num_intervals)  # Reduced from 200
    Model.FieldOutputRequest(name='F-Output-2', createStepName=step_name, variables=('LE','S','SDEG','STATUS'))
    Model.fieldOutputRequests['F-Output-2'].setValues(numIntervals=num_intervals)  # Reduced from 200

    # OPTIMIZED: Simplified BCs
    W_x = 2.0 * Nx * L
    W_y = Ny * math.sqrt(3.0) * L
    rp = asm.ReferencePoint(point=((W_x - L)/2.0, W_y - math.sqrt(3.0)*L/2.0, 0.0))
    Ref1 = asm.referencePoints[rp.id]
    asm.Set(referencePoints=(Ref1,), name='Ref1')
    regionRP = asm.sets['Ref1']
    Model.HistoryOutputRequest(name='Ref1', createStepName=step_name, variables=('U2','RF2'), region=regionRP).setValues(numIntervals=num_intervals)

    # Simplified tie constraint
    tie_nodes = inst.nodes.getByBoundingBox(
        xMin=-L/2.0, yMin=W_y - math.sqrt(3.0)*L/2.0 - el_size/10.0, zMin=-1.0,
        xMax=W_x,     yMax=W_y + el_size/10.0,                          zMax= 1.0
    )
    tieRegion = regionToolset.Region(nodes=tie_nodes)
    Model.RigidBody(name='Constraint-1', refPointRegion=regionRP, tieRegion=tieRegion)
    Model.VelocityBC(name='Up', createStepName=step_name, region=regionRP, v1=0.0, v2=Vy)

    # Simplified bottom BC
    bottom_nodes = inst.nodes.getByBoundingBox(
        xMin=-L/2.0 - el_size,   yMin=-L*math.sqrt(3.0)/2.0 - el_size, zMin=-1.0,
        xMax=W_x + el_size,      yMax=-L*math.sqrt(3.0)/2.0 + el_size, zMax= 1.0
    )
    botRegion = regionToolset.Region(nodes=bottom_nodes)
    Model.DisplacementBC(name='Bottom', createStepName=step_name, region=botRegion, u1=0.0, u2=0.0)

    # Skip LR BC for speed
    # LR_nodes = inst.nodes.getByBoundingBox(...)
    # Model.DisplacementBC(name='Left-and-right', ...)

    # Job
    mdb.Job(name=job_name, model=model_name, numCpus=6, numDomains=6).writeInput()
    
    t7 = time.time()
    print('  BCs & Job: %.2fs' % (t7-t6))
    print('  TOTAL: %.2fs' % (t7-t0))
    print('Built %s, job %s' % (model_name, job_name))


def main():
    print('Starting OPTIMIZED Kagome parametric grid...')
    print('Optimizations: Direct geometry, simple sets, no dislocation, reduced outputs, minimal BCs')
    print('='*70)
    
    total_start = time.time()
    idx = 1
    total_models = len(ru_list) * len(removal_fraction_list) * len(max_r_mult_list) * len(thickness_var_pct_list)
    
    for ru in ru_list:
        for rem in removal_fraction_list:
            for mr in max_r_mult_list:
                for var in thickness_var_pct_list:
                    print('\n--- Model %d/%d ---' % (idx, total_models))
                    build_one(ru, rem, mr, var, idx)
                    idx += 1
    
    total_time = time.time() - total_start
    print('\n' + '='*70)
    print('OPTIMIZED Parametric grid complete!')
    print('Total models: %d' % total_models)
    print('Total time: %.2f seconds' % total_time)
    print('Average per model: %.2f seconds' % (total_time / total_models))
    print('='*70)


if __name__ == '__main__':
    main()


