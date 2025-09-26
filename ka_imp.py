# -*- coding: utf-8 -*-
"""
Final refactored script for Kagome lattice toughness
- Builds a 2D Kagome lattice
- Converts to orphan mesh
- Randomly deletes a fraction of bars
- Reassigns beam sections and boundary conditions to the orphan instance
- Writes Abaqus job input
Note: uses Python 2.7-compatible string formatting (no f-strings)
"""

from abaqus import *
from abaqusConstants import *
import job, mesh, part, regionToolset, sketch
import math, random
import time
t0 = time.time()  # Start timing
#from abaqusConstants import MAX_PRINCIPAL_STRESS

# === GLOBAL PARAMETERS ===
L = 1.00                       # Bar length [mm]
el_size = L/5                 # Mesh element size [mm]
#W = 100.0                     # Sample length [mm]
Bi = 1                         # Sample width [mm]
p = 0.05                       # Relative density
t = p * L / math.sqrt(3.0)     # Bar thickness [mm]
Vy = 10.0                       # Velocity [mm/s]

Nx = 50                        # Number of unit cells in x-direction
Ny = int(round(11.0 / math.sqrt(3.0) * Nx)/2.00)  # Number of unit cells in y-direction
failure_strain = 0.0015          # Failure strain
Period = failure_strain * int(round(11.0 / math.sqrt(3.0) * Nx)/2.00) * math.sqrt(3.0) * L / Vy            # Time period [s] = strian length / velocity
max_r = 0.0 * L  # maximum radial displacement
material_type = 'Brittle'     # Either 'Brittle' or 'Ductile'
# Random removal settings
removal_fraction = 0.0003       # Fraction of bars to remove

# Parameters
#Nx = 10
#Ny = 30
#L = 1.0
# Constants
dx = 2.0 * L
dy = math.sqrt(3) * L
dy_vertex = math.sqrt(3)/2 * L

# Flat-topped hexagon offsets
hex_offsets = [
    ( L, 0),
    ( 0.5*L,  dy_vertex),
    (-0.5*L,  dy_vertex),
    (-L, 0),
    (-0.5*L, -dy_vertex),
    ( 0.5*L, -dy_vertex)
]


# Material properties
ps = 7.8e-9                   # Density [g/mm^3]
Es = 200000                   # Young's modulus [MPa]
S_0s = 200                 # Yield strength
vs = 0.3                      # Poisson's ratio
Ef = 1e-6                    # Failure strain
Uf = Ef * el_size             # Failure displacement [mm]
nbr_frames = 200              # Frames for field outputs


#random_seed = 123             # Seed for reproducibility



# Loop over lattice sizes
for j in [1]:

    
    # Derived dimensions
    #Nx = int(j)
    Ny = int(round(11.0 / math.sqrt(3.0) * Nx)/2.00)
    W_x = 2 * Nx  * L
    W_y = Ny  * math.sqrt(3.0) * L
    # --- Create model and part ---
    
    model_name = 'Kagome%d' % Nx
    Model = mdb.Model(name=model_name)
    part_name = 'Kagome%d' % j
    Part = Model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    
    # --- Sketch unit cell ---
    sheetSize = max(2.1 * Nx * 2 * L, 2.1 * Ny * 2 * math.sqrt(3.0) * L)
    sketch = Model.ConstrainedSketch(name='Kagome%d' % j, sheetSize=sheetSize)
    # Global map of nodes
    pos_to_node = {}
    
    # Displacement using radial random angle
    def displace_radially(x, y):
        theta = random.uniform(0.000, 2.000 * math.pi)
        r = random.uniform(0.000, 1.000) * max_r
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)
        return (x + dx, y + dy)
    
    # Create edges from tessellated hexagons
    edges = []
    for jj in range(Ny):
        cy = jj * dy
        x_offset = 0 if jj % 2 == 0 else L
        for i in range(Nx):
            cx = i * dx + x_offset
            hex_points = []
            for offset in hex_offsets:
                px = cx + offset[0]
                py = cy + offset[1]
                key = (round(px, 6), round(py, 6))
    
                if key not in pos_to_node:
                    pos_to_node[key] = displace_radially(px, py)
    
                hex_points.append(pos_to_node[key])
    
            # Create 6 edges of the hexagon
            for k in range(6):
                p1 = hex_points[k]
                p2 = hex_points[(k + 1) % 6]
                edges.append((p1, p2))
    
    # Draw all lines
    for pt1, pt2 in edges:
        sketch.Line(point1=pt1, point2=pt2)
    
    # Create wire part

    Part.BaseWire(sketch=sketch)


    
    # --- Material and section ---
    mat = Model.Material(name='damage')
    mat.Density(table=((ps,),))
    mat.Elastic(table=((Es, vs),))
    if material_type == 'Ductile':
        mat.DuctileDamageInitiation(table=((Ef, 0.0, 0.0),))
        mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((Uf,),), viscosity=1.0e-6)
        
        mat.Plastic(table=(
            (34.6179,0.0),(35.439,0.000212981),(36.26,0.00044262),(37.0811,0.000700668),
            (37.9022,0.000994008),(38.7232,0.00131721),(39.5443,0.00167037),(40.3654,0.00207179),
            (41.1864,0.00251601),(42.0075,0.00303611),(42.8286,0.00362178),(43.6496,0.00427991),
            (44.4707,0.00505844),(45.2917,0.00594553),(46.1128,0.00698035),(46.9339,0.00821897),
            (47.7549,0.00975866),(48.576,0.0116222),(49.3967,0.0141484),(50.2118,0.0178363),
            (50.9537,0.023326),(51.5097,0.0300158),(51.8997,0.037387),(52.1912,0.0445882),
            (52.5259,0.0520426),(52.9769,0.0591642),(53.4702,0.0656527)
        ))
    elif material_type == 'Brittle':
        mat.DuctileDamageInitiation(table=((Ef, 0.0, 0.0),))
        mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((Uf,),))
        
        mat.Plastic(table=(
            (S_0s,0.0),
        ))
    else:
        print('no material available')

    Model.RectangularProfile(name='Rect', a=Bi, b=t)
    Model.BeamSection(name='BeamSection', integration=DURING_ANALYSIS, profile='Rect', material='damage')

    # Assign section on native mesh for orphan conversion
    reg = Part.Set(name='EntireLatticeSet', edges=Part.edges)
    Part.SectionAssignment(region=reg, sectionName='BeamSection')
    Part.assignBeamSectionOrientation(region=reg, method=N1_COSINES, n1=(0.0,0.0,1.0))

    # --- Mesh ---
    Part.seedPart(size=el_size)
    Part.setElementType(regions=reg, elemTypes=(mesh.ElemType(elemCode=B21),))
    Part.generateMesh()

    # --- Create instance and convert to orphan mesh ---
    asm = Model.rootAssembly
    nativeInst = asm.Instance(name='Inst_native', part=Part, dependent=ON)
    orphan_name = 'OrphanKagome%d' % j
    orphanPart = asm.PartFromInstanceMesh(name=orphan_name, partInstances=(nativeInst,))

    # --- Random deletion of elements ---
    #random.seed(random_seed)
    elems = list(orphanPart.elements)
    n_remove = int(len(elems) * removal_fraction)
    to_remove = random.sample(elems, n_remove)
    if to_remove:
        print('Removing %d elements...' % len(to_remove))
        orphanPart.deleteElement(elements=tuple(to_remove), deleteUnreferencedNodes=ON)
    else:
        print('No elements removed. Mesh remains complete.')


    # --- Reassign section to remaining bars ---
    elemRegion = regionToolset.Region(elements=orphanPart.elements)
    orphanPart.SectionAssignment(region=elemRegion, sectionName='BeamSection')
    orphanPart.assignBeamSectionOrientation(region=elemRegion, method=N1_COSINES, n1=(0.0,0.0,1.0))

    # --- Swap in orphan instance and fetch it ---
    asm.features.changeKey(fromName='Inst_native', toName='Inst_old')
    asm.Instance(name='Inst', part=orphanPart, dependent=ON)
    inst = asm.instances['Inst']
    asm.features['Inst_old'].suppress()

    # --- Define step and field outputs ---
    step_name = 'damage'
    Model.ExplicitDynamicsStep(name=step_name, previous='Initial', timePeriod=Period, improvedDtMethod=ON)
    Model.FieldOutputRequest(name='F-Output-1', createStepName=step_name, variables=('RF','U'))
    Model.fieldOutputRequests['F-Output-1'].setValues(numIntervals=nbr_frames)
    Model.FieldOutputRequest(name='F-Output-2', createStepName=step_name, variables=('LE','S','SDEG','STATUS'))
    Model.fieldOutputRequests['F-Output-2'].setValues(numIntervals=nbr_frames)




    # or scoped:
    # kb.insert(-1, '*DAMPING, REGION=LATTICE, ALPHA=4.6565e5, BETA=4.45e-11')


    # --- Set mass scaling ---
    #Model.steps['damage'].setValues(massScaling=((
    #    SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 0.0, 5e-07, BELOW_MIN, 0, 0, 0.0, 
    #    0.0, 0, None), ))

    # --- Create reference point, sets, and history ---
    rp = asm.ReferencePoint(point=((W_x-L)/2, W_y-math.sqrt(3.0)*L/2.00, 0))
    Ref1 = asm.referencePoints[rp.id]
    #regionRP = regionToolset.Region(referencePoints=(Ref1,)) 
    #asm.Set(referencePoints=(Ref1,), name='Ref1')
    #Model.HistoryOutputRequest(name='Ref1', createStepName=step_name, variables=('U2','RF2'), region=regionRP).setValues(numIntervals=nbr_frames)



    # ...existing code...
    asm.Set(referencePoints=(Ref1,), name='Ref1')
    regionRP = asm.sets['Ref1']
    Model.HistoryOutputRequest(name='Ref1', createStepName=step_name, variables=('U2','RF2'), region=regionRP).setValues(numIntervals=nbr_frames)
    # ...existing code...

    
    # --- Rigid body tie and BCs on orphan inst ---
    # Tie the top nodes to the reference point via a rigid body
    tie_nodes = inst.nodes.getByBoundingBox(
        xMin=-L/2.00-max_r, yMin=W_y-L*math.sqrt(3.00)/2.00-max_r-el_size/10, zMin=-1.0,
          xMax=W_x+max_r, yMax=W_y+max_r+el_size/10, zMax=1.0
    )
    tieRegion = regionToolset.Region(nodes=tie_nodes)
    Model.RigidBody(
        name='Constraint-1', refPointRegion=regionRP,
        tieRegion=tieRegion
    )

    # Apply upward velocity to the reference point
    Model.VelocityBC(
        name='Up', createStepName=step_name, region=regionRP,
        #v1=0.0, v2=Vy, vr3=0.0
        v2=Vy
    )

    # Constrain bottom nodes in both directions
    bottom_nodes = inst.nodes.getByBoundingBox(
        xMin=-L/2 - max_r - el_size/1.0,
        yMin=-L*math.sqrt(3.0)/2.00 - max_r - el_size/1.0,
        xMax= el_size/1.0 + W_x + max_r,
        yMax=-L*math.sqrt(3.0)/2.00 + max_r + el_size/1.0
    )
    botRegion = regionToolset.Region(nodes=bottom_nodes)
    Model.DisplacementBC(
        name='Bottom', createStepName=step_name,
        region=botRegion, u2=0.0
    )

    # Constrain left and right nodes in both directions
    LR_nodes = inst.nodes.getByBoundingBox(
        xMin=-L - max_r,
        yMin=-L*math.sqrt(3.0)/2.00 - max_r,
        xMax= -L + max_r,
        yMax=W_y-L*math.sqrt(3.0)/2.00 + max_r
    )
    #+ inst.nodes.getByBoundingBox(
    #    xMin=W_x - max_r,
    #    yMin=-L*math.sqrt(3.0)/2.00 - max_r,
    #    xMax= W_x + max_r,
    #    yMax=W_y-L*math.sqrt(3.0)/2.00 + max_r
    #)

    botRegion = regionToolset.Region(nodes=LR_nodes)
    Model.DisplacementBC(
        name='Left-and-right', createStepName=step_name,
        region=botRegion, u1=0.0
    )


    
    # --- Write job input ---
    print('Writing job input...')
    print('Lattice size: W = %.3f L = %.3f' % (W_x, W_y))
    ru_token = ('%.4f' % p).replace('.', 'p')
    del_token = ('%.4f' % removal_fraction).replace('.', 'p')
    dis_token = ('%.4f' % max_r).replace('.', 'p')
    job_name = 'UNI_Kagome_ru%s_dis%s_del%s_Nx%d' % (ru_token, dis_token, del_token, Nx)
    mdb.Job(name=job_name, model=model_name, numCpus=6, numDomains=6).writeInput()

# --- Print elapsed time ---
t1 = time.time()
elapsed = t1 - t0
print("Elapsed time: %.2f seconds" % elapsed)
