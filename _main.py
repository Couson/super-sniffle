import pyvista as pv
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============== TRANSFORM UTILITIES ==============

def apply_transforms(mesh, center, rotation=None, scale=None):
    """Apply rotation and scale transforms to a mesh."""
    if scale:
        if isinstance(scale, (int, float)):
            mesh = mesh.scale([scale, scale, scale], inplace=False)
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            mesh = mesh.scale(scale, inplace=False)
    if rotation:
        rx = rotation.get('x', 0) if isinstance(rotation, dict) else rotation[0] if len(rotation) > 0 else 0
        ry = rotation.get('y', 0) if isinstance(rotation, dict) else rotation[1] if len(rotation) > 1 else 0
        rz = rotation.get('z', 0) if isinstance(rotation, dict) else rotation[2] if len(rotation) > 2 else 0
        if rx: mesh = mesh.rotate_x(rx, point=center, inplace=False)
        if ry: mesh = mesh.rotate_y(ry, point=center, inplace=False)
        if rz: mesh = mesh.rotate_z(rz, point=center, inplace=False)
    return mesh

# ============== BASIC PRIMITIVES ==============

def create_box(width=1, height=1, depth=1, center=(0, 0, 0), rotation=None, scale=None, resolution=1):
    """Create a box mesh with optional subdivisions."""
    mesh = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2], center[2] + height
    ))
    if resolution > 1:
        mesh = mesh.subdivide(resolution - 1)
    return apply_transforms(mesh, center, rotation, scale)

def create_rounded_box(width=1, height=1, depth=1, radius=0.1, center=(0, 0, 0), rotation=None, scale=None):
    """Create a box with rounded edges."""
    # Create base box slightly smaller
    r = min(radius, width/4, height/4, depth/4)
    mesh = pv.Box(bounds=(
        center[0] - width/2 + r, center[0] + width/2 - r,
        center[1] - depth/2 + r, center[1] + depth/2 - r,
        center[2] + r, center[2] + height - r
    ))
    # Expand with smoothing
    mesh = mesh.extract_surface()
    mesh = mesh.smooth(n_iter=50)
    return apply_transforms(mesh, center, rotation, scale)

def create_cylinder(radius=1, height=1, center=(0, 0, 0), rotation=None, scale=None, resolution=40, capped=True):
    """Create a cylinder mesh."""
    c = (center[0], center[1], center[2] + height/2)
    mesh = pv.Cylinder(radius=radius, height=height, center=c, direction=(0, 0, 1), 
                       resolution=resolution, capping=capped)
    return apply_transforms(mesh, center, rotation, scale)

def create_cone(radius=1, height=1, center=(0, 0, 0), rotation=None, scale=None, resolution=40):
    """Create a cone mesh."""
    c = (center[0], center[1], center[2] + height/2)
    mesh = pv.Cone(radius=radius, height=height, center=c, direction=(0, 0, 1), resolution=resolution)
    return apply_transforms(mesh, center, rotation, scale)

def create_sphere(radius=1, center=(0, 0, 0), rotation=None, scale=None, resolution=20):
    """Create a sphere mesh with adjustable resolution."""
    mesh = pv.Sphere(radius=radius, center=(center[0], center[1], center[2] + radius),
                     theta_resolution=resolution, phi_resolution=resolution)
    return apply_transforms(mesh, center, rotation, scale)

def create_hemisphere(radius=1, center=(0, 0, 0), rotation=None, scale=None, resolution=20):
    """Create a hemisphere (half sphere)."""
    mesh = pv.Sphere(radius=radius, center=(center[0], center[1], center[2]),
                     theta_resolution=resolution, phi_resolution=resolution,
                     start_theta=0, end_theta=180)
    return apply_transforms(mesh, center, rotation, scale)

def create_ellipsoid(radius_x=1, radius_y=1, radius_z=1, center=(0, 0, 0), rotation=None, scale=None):
    """Create an ellipsoid."""
    mesh = pv.ParametricEllipsoid(radius_x, radius_y, radius_z)
    mesh = mesh.translate((center[0], center[1], center[2] + radius_z), inplace=False)
    return apply_transforms(mesh, center, rotation, scale)

def create_plane(width=10, depth=10, center=(0, 0, 0), rotation=None, scale=None, resolution=1):
    """Create a flat plane with subdivisions for terrain."""
    mesh = pv.Plane(center=center, direction=(0, 0, 1), i_size=width, j_size=depth,
                    i_resolution=max(1, resolution), j_resolution=max(1, resolution))
    return apply_transforms(mesh, center, rotation, scale)

def create_terrain(width=50, depth=50, center=(0, 0, 0), roughness=3, seed=None, max_height=5, resolution=50):
    """Create procedural terrain with hills."""
    if seed:
        np.random.seed(seed)
    mesh = pv.Plane(center=(center[0], center[1], center[2]), direction=(0, 0, 1),
                    i_size=width, j_size=depth, i_resolution=resolution, j_resolution=resolution)
    # Add noise for hills
    points = mesh.points.copy()
    for _ in range(roughness):
        freq = np.random.uniform(0.05, 0.2)
        amp = np.random.uniform(0.3, 1.0) * max_height
        phase = np.random.uniform(0, 2 * np.pi, 2)
        points[:, 2] += amp * np.sin(freq * points[:, 0] + phase[0]) * np.cos(freq * points[:, 1] + phase[1])
    points[:, 2] = np.maximum(points[:, 2], center[2])  # Keep above ground
    mesh.points = points
    return mesh

def create_pyramid(base=1, height=1, sides=4, center=(0, 0, 0), rotation=None, scale=None):
    """Create a pyramid with configurable sides."""
    angles = np.linspace(0, 2 * np.pi, sides + 1)[:-1] + np.pi/sides
    half = base / 2
    # Base points
    points = [[center[0] + half * np.cos(a), center[1] + half * np.sin(a), center[2]] for a in angles]
    points.append([center[0], center[1], center[2] + height])  # Apex
    points = np.array(points)
    
    # Faces
    faces = [[sides] + list(range(sides))]  # Base
    for i in range(sides):
        faces.append([3, i, (i + 1) % sides, sides])  # Side triangles
    faces = np.hstack(faces)
    mesh = pv.PolyData(points, faces)
    return apply_transforms(mesh, center, rotation, scale)

def create_prism(radius=1, height=1, sides=6, center=(0, 0, 0), rotation=None, scale=None):
    """Create a prism (hexagon, pentagon, etc.)."""
    angles = np.linspace(0, 2 * np.pi, sides + 1)[:-1]
    # Bottom and top vertices
    bottom = [[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), center[2]] for a in angles]
    top = [[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), center[2] + height] for a in angles]
    points = np.array(bottom + top)
    
    # Faces
    faces = []
    faces.append([sides] + list(range(sides)))  # Bottom
    faces.append([sides] + list(range(sides, 2 * sides)))  # Top
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([4, i, j, j + sides, i + sides])  # Sides
    faces = np.hstack(faces)
    mesh = pv.PolyData(points, faces)
    return apply_transforms(mesh, center, rotation, scale)

def create_torus(ring_radius=2, tube_radius=0.5, center=(0, 0, 0), rotation=None, scale=None, resolution=40):
    """Create a torus (donut)."""
    mesh = pv.ParametricTorus(ringradius=ring_radius, crosssectionradius=tube_radius)
    mesh = mesh.translate((center[0], center[1], center[2] + tube_radius), inplace=False)
    return apply_transforms(mesh, center, rotation, scale)

def create_disc(radius=1, inner_radius=0, center=(0, 0, 0), rotation=None, scale=None, resolution=40):
    """Create a disc (optionally with hole)."""
    mesh = pv.Disc(center=center, normal=(0, 0, 1), inner=inner_radius, outer=radius, c_res=resolution)
    return apply_transforms(mesh, center, rotation, scale)

def create_tube(radius=1, height=1, thickness=0.2, center=(0, 0, 0), rotation=None, scale=None, resolution=40):
    """Create a hollow tube."""
    outer = pv.Cylinder(radius=radius, height=height, 
                        center=(center[0], center[1], center[2] + height/2),
                        direction=(0, 0, 1), resolution=resolution, capping=False)
    top_cap = pv.Disc(center=(center[0], center[1], center[2] + height),
                      normal=(0, 0, 1), inner=radius - thickness, outer=radius, c_res=resolution)
    bottom_cap = pv.Disc(center=(center[0], center[1], center[2]),
                         normal=(0, 0, -1), inner=radius - thickness, outer=radius, c_res=resolution)
    inner = pv.Cylinder(radius=radius - thickness, height=height,
                        center=(center[0], center[1], center[2] + height/2),
                        direction=(0, 0, 1), resolution=resolution, capping=False)
    mesh = outer + top_cap + bottom_cap + inner
    return apply_transforms(mesh, center, rotation, scale)

def create_arch(width=2, height=2, depth=1, thickness=0.3, center=(0, 0, 0), rotation=None, scale=None):
    """Create an arch."""
    mesh = pv.ParametricTorus(ringradius=width/2, crosssectionradius=thickness)
    mesh = mesh.clip(normal=(0, 0, -1), origin=(0, 0, 0), inplace=False)
    mesh = mesh.scale([1, depth/thickness/2, height/width], inplace=False)
    mesh = mesh.translate((center[0], center[1], center[2]), inplace=False)
    return apply_transforms(mesh, center, rotation, scale)

def create_stairs(width=2, height=2, depth=4, steps=4, center=(0, 0, 0), rotation=None, scale=None):
    """Create stairs."""
    meshes = []
    step_height = height / steps
    step_depth = depth / steps
    for i in range(steps):
        step = pv.Box(bounds=(
            center[0] - width/2, center[0] + width/2,
            center[1] + i * step_depth, center[1] + (i + 1) * step_depth,
            center[2], center[2] + (i + 1) * step_height
        ))
        meshes.append(step)
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return apply_transforms(mesh, center, rotation, scale)

def create_wedge(width=1, height=1, depth=1, center=(0, 0, 0), rotation=None, scale=None):
    """Create a wedge/ramp."""
    points = np.array([
        [center[0] - width/2, center[1] - depth/2, center[2]],
        [center[0] + width/2, center[1] - depth/2, center[2]],
        [center[0] + width/2, center[1] + depth/2, center[2]],
        [center[0] - width/2, center[1] + depth/2, center[2]],
        [center[0] - width/2, center[1] - depth/2, center[2] + height],
        [center[0] - width/2, center[1] + depth/2, center[2] + height],
    ])
    faces = np.hstack([
        [4, 0, 3, 2, 1],  # bottom
        [4, 0, 4, 5, 3],  # left
        [3, 0, 1, 4],     # front
        [3, 3, 5, 2],     # back
        [4, 1, 2, 5, 4],  # ramp
    ])
    mesh = pv.PolyData(points, faces)
    return apply_transforms(mesh, center, rotation, scale)

# ============== COMPOUND SHAPES ==============

def create_tree(trunk_radius=0.5, trunk_height=4, crown_radius=3, crown_height=6, 
                crown_type="cone", center=(0, 0, 0), rotation=None, scale=None):
    """Create a tree with trunk and crown."""
    trunk = pv.Cylinder(radius=trunk_radius, height=trunk_height,
                        center=(center[0], center[1], center[2] + trunk_height/2),
                        direction=(0, 0, 1), resolution=20)
    if crown_type == "cone":
        crown = pv.Cone(radius=crown_radius, height=crown_height,
                        center=(center[0], center[1], center[2] + trunk_height + crown_height/2),
                        direction=(0, 0, 1), resolution=20)
    elif crown_type == "sphere":
        crown = pv.Sphere(radius=crown_radius,
                          center=(center[0], center[1], center[2] + trunk_height + crown_radius))
    else:  # layered cones
        crown = pv.Cone(radius=crown_radius, height=crown_height * 0.5,
                        center=(center[0], center[1], center[2] + trunk_height + crown_height * 0.25),
                        direction=(0, 0, 1))
        crown2 = pv.Cone(radius=crown_radius * 0.7, height=crown_height * 0.4,
                         center=(center[0], center[1], center[2] + trunk_height + crown_height * 0.5),
                         direction=(0, 0, 1))
        crown = crown + crown2
    mesh = trunk + crown
    return apply_transforms(mesh, center, rotation, scale)

def create_wall(width=10, height=5, thickness=0.5, 
                windows=None, doors=None, center=(0, 0, 0), rotation=None, scale=None):
    """Create a wall with optional window/door cutouts (visual only)."""
    # Main wall
    wall = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - thickness/2, center[1] + thickness/2,
        center[2], center[2] + height
    ))
    mesh = wall
    return apply_transforms(mesh, center, rotation, scale)

def create_roof(width=10, depth=10, height=4, style="gable", center=(0, 0, 0), rotation=None, scale=None):
    """Create different roof styles."""
    if style == "pyramid" or style == "hip":
        mesh = create_pyramid(base=max(width, depth) * 1.1, height=height, sides=4, center=center)
    elif style == "dome":
        mesh = create_hemisphere(radius=min(width, depth)/2, center=center)
        mesh = mesh.scale([width/min(width,depth), depth/min(width,depth), height*2/min(width,depth)], inplace=False)
    else:  # gable
        points = np.array([
            [center[0] - width/2, center[1] - depth/2, center[2]],
            [center[0] + width/2, center[1] - depth/2, center[2]],
            [center[0] + width/2, center[1] + depth/2, center[2]],
            [center[0] - width/2, center[1] + depth/2, center[2]],
            [center[0], center[1] - depth/2, center[2] + height],
            [center[0], center[1] + depth/2, center[2] + height],
        ])
        faces = np.hstack([
            [3, 0, 1, 4],  # front
            [3, 2, 3, 5],  # back
            [4, 0, 4, 5, 3],  # left slope
            [4, 1, 2, 5, 4],  # right slope
            [4, 0, 3, 2, 1],  # bottom
        ])
        mesh = pv.PolyData(points, faces)
    return apply_transforms(mesh, center, rotation, scale)

def create_fence(length=20, height=2, post_width=0.2, post_spacing=2, rail_height=0.1,
                 center=(0, 0, 0), rotation=None, scale=None):
    """Create a fence with posts and rails."""
    meshes = []
    n_posts = int(length / post_spacing) + 1
    for i in range(n_posts):
        x = center[0] - length/2 + i * post_spacing
        post = pv.Box(bounds=(
            x - post_width/2, x + post_width/2,
            center[1] - post_width/2, center[1] + post_width/2,
            center[2], center[2] + height
        ))
        meshes.append(post)
    # Rails
    for h in [height * 0.3, height * 0.7]:
        rail = pv.Box(bounds=(
            center[0] - length/2, center[0] + length/2,
            center[1] - rail_height/2, center[1] + rail_height/2,
            center[2] + h - rail_height/2, center[2] + h + rail_height/2
        ))
        meshes.append(rail)
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return apply_transforms(mesh, center, rotation, scale)

def create_column(radius=0.5, height=6, base_height=0.5, capital_height=0.5,
                  center=(0, 0, 0), rotation=None, scale=None, style="doric"):
    """Create a classical column."""
    # Base
    base = pv.Cylinder(radius=radius * 1.3, height=base_height,
                       center=(center[0], center[1], center[2] + base_height/2),
                       direction=(0, 0, 1))
    # Shaft
    shaft = pv.Cylinder(radius=radius, height=height - base_height - capital_height,
                        center=(center[0], center[1], center[2] + base_height + (height - base_height - capital_height)/2),
                        direction=(0, 0, 1))
    # Capital
    capital = pv.Cylinder(radius=radius * 1.4, height=capital_height,
                          center=(center[0], center[1], center[2] + height - capital_height/2),
                          direction=(0, 0, 1))
    mesh = base + shaft + capital
    return apply_transforms(mesh, center, rotation, scale)

# ============== ARRAY/PATTERN ==============

def create_array(base_type, count_x=1, count_y=1, count_z=1, spacing_x=2, spacing_y=2, spacing_z=2,
                 center=(0, 0, 0), rotation=None, scale=None, **kwargs):
    """Create an array of shapes."""
    if base_type not in SHAPES:
        return pv.Sphere(radius=1, center=center)  # Fallback
    
    meshes = []
    for i in range(count_x):
        for j in range(count_y):
            for k in range(count_z):
                offset = (
                    center[0] + i * spacing_x - (count_x - 1) * spacing_x / 2,
                    center[1] + j * spacing_y - (count_y - 1) * spacing_y / 2,
                    center[2] + k * spacing_z
                )
                # Remove array-specific params before passing to base shape
                base_kwargs = {key: val for key, val in kwargs.items() 
                              if key not in ['count_x', 'count_y', 'count_z', 
                                            'spacing_x', 'spacing_y', 'spacing_z', 'base_type']}
                base_kwargs['center'] = offset
                m = SHAPES[base_type](**base_kwargs)
                meshes.append(m)
    
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return apply_transforms(mesh, center, rotation, scale)

def create_circular_array(base_type, count=6, radius=5, center=(0, 0, 0), 
                          rotation=None, scale=None, **kwargs):
    """Create shapes arranged in a circle."""
    if base_type not in SHAPES:
        return pv.Sphere(radius=1, center=center)
    
    meshes = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        offset = (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2]
        )
        base_kwargs = {key: val for key, val in kwargs.items() 
                      if key not in ['count', 'base_type']}
        base_kwargs['center'] = offset
        base_kwargs['rotation'] = {'z': np.degrees(angle)}
        m = SHAPES[base_type](**base_kwargs)
        meshes.append(m)
    
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return mesh

# ============== BUILDINGS ==============

def create_house(width=10, height=8, depth=10, roof_height=4, roof_style="gable",
                 center=(0, 0, 0), rotation=None, scale=None):
    """Create a simple house with roof."""
    # Main body
    body = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2], center[2] + height
    ))
    # Roof
    roof = create_roof(width=width*1.1, depth=depth*1.1, height=roof_height, 
                       style=roof_style, center=(center[0], center[1], center[2] + height))
    # Door
    door = pv.Box(bounds=(
        center[0] - width*0.1, center[0] + width*0.1,
        center[1] - depth/2 - 0.1, center[1] - depth/2 + 0.1,
        center[2], center[2] + height*0.45
    ))
    mesh = body + roof + door
    return apply_transforms(mesh, center, rotation, scale)

def create_tower(radius=3, height=15, roof_height=5, crenellations=True,
                 center=(0, 0, 0), rotation=None, scale=None):
    """Create a castle tower."""
    # Main cylinder
    body = pv.Cylinder(radius=radius, height=height,
                       center=(center[0], center[1], center[2] + height/2),
                       direction=(0, 0, 1), resolution=32)
    # Conical roof
    roof = pv.Cone(radius=radius * 1.2, height=roof_height,
                   center=(center[0], center[1], center[2] + height + roof_height/2),
                   direction=(0, 0, 1), resolution=32)
    mesh = body + roof
    
    # Add crenellations (battlements)
    if crenellations:
        n_cren = 8
        for i in range(n_cren):
            angle = 2 * np.pi * i / n_cren
            x = center[0] + radius * 0.85 * np.cos(angle)
            y = center[1] + radius * 0.85 * np.sin(angle)
            cren = pv.Box(bounds=(x - 0.3, x + 0.3, y - 0.3, y + 0.3, 
                                  center[2] + height, center[2] + height + 1))
            mesh = mesh + cren
    
    return apply_transforms(mesh, center, rotation, scale)

def create_castle(width=30, depth=30, wall_height=10, tower_height=18, 
                  center=(0, 0, 0), rotation=None, scale=None):
    """Create a castle with walls and corner towers."""
    meshes = []
    
    # Four corner towers
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        tower = create_tower(radius=3, height=tower_height, roof_height=5,
                            center=(center[0] + dx*width/2, center[1] + dy*depth/2, center[2]))
        meshes.append(tower)
    
    # Walls between towers
    wall_thickness = 2
    # Front and back walls
    for dy in [-depth/2, depth/2]:
        wall = pv.Box(bounds=(
            center[0] - width/2 + 3, center[0] + width/2 - 3,
            center[1] + dy - wall_thickness/2, center[1] + dy + wall_thickness/2,
            center[2], center[2] + wall_height
        ))
        meshes.append(wall)
    # Side walls
    for dx in [-width/2, width/2]:
        wall = pv.Box(bounds=(
            center[0] + dx - wall_thickness/2, center[0] + dx + wall_thickness/2,
            center[1] - depth/2 + 3, center[1] + depth/2 - 3,
            center[2], center[2] + wall_height
        ))
        meshes.append(wall)
    
    # Gate
    gate = create_arch(width=4, height=6, depth=wall_thickness + 0.5, thickness=1,
                       center=(center[0], center[1] - depth/2, center[2]))
    meshes.append(gate)
    
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return apply_transforms(mesh, center, rotation, scale)

def create_church(width=12, depth=20, height=15, tower_height=25, 
                  center=(0, 0, 0), rotation=None, scale=None):
    """Create a church with steeple."""
    # Main nave
    nave = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2], center[2] + height
    ))
    # Roof
    roof = create_roof(width=width*1.1, depth=depth*1.1, height=height*0.5,
                       style="gable", center=(center[0], center[1], center[2] + height))
    # Tower/steeple
    tower_w = width * 0.4
    tower = pv.Box(bounds=(
        center[0] - tower_w/2, center[0] + tower_w/2,
        center[1] + depth/2 - tower_w, center[1] + depth/2,
        center[2], center[2] + tower_height
    ))
    # Steeple
    steeple = pv.Cone(radius=tower_w * 0.6, height=tower_height * 0.3,
                      center=(center[0], center[1] + depth/2 - tower_w/2, 
                              center[2] + tower_height + tower_height*0.15),
                      direction=(0, 0, 1))
    
    mesh = nave + roof + tower + steeple
    return apply_transforms(mesh, center, rotation, scale)

def create_skyscraper(width=10, depth=10, height=50, floors=15,
                      center=(0, 0, 0), rotation=None, scale=None):
    """Create a modern skyscraper."""
    # Main building
    body = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2], center[2] + height
    ))
    # Floor lines (visual detail)
    mesh = body
    floor_height = height / floors
    for i in range(1, floors):
        line = pv.Box(bounds=(
            center[0] - width/2 - 0.1, center[0] + width/2 + 0.1,
            center[1] - depth/2 - 0.1, center[1] + depth/2 + 0.1,
            center[2] + i * floor_height - 0.1, center[2] + i * floor_height + 0.1
        ))
        mesh = mesh + line
    
    return apply_transforms(mesh, center, rotation, scale)

def create_windmill(tower_height=12, blade_length=8, center=(0, 0, 0), 
                    rotation=None, scale=None):
    """Create a windmill."""
    # Tower (tapered)
    tower_bottom = 4
    tower_top = 2
    # Approximate tapered tower with cylinder
    tower = pv.Cylinder(radius=tower_bottom/2, height=tower_height,
                        center=(center[0], center[1], center[2] + tower_height/2),
                        direction=(0, 0, 1))
    # Roof
    roof = pv.Cone(radius=tower_top, height=3,
                   center=(center[0], center[1], center[2] + tower_height + 1.5),
                   direction=(0, 0, 1))
    # Blades hub
    hub = pv.Sphere(radius=0.5, center=(center[0], center[1] - tower_bottom/2 - 0.5, 
                                        center[2] + tower_height * 0.8))
    # Four blades
    mesh = tower + roof + hub
    for angle in [0, 90, 180, 270]:
        blade = pv.Box(bounds=(-0.3, 0.3, -blade_length/2, blade_length/2, -0.1, 0.1))
        blade = blade.rotate_z(angle, inplace=False)
        blade = blade.rotate_x(15, inplace=False)
        blade = blade.translate((center[0], center[1] - tower_bottom/2 - 1, 
                                 center[2] + tower_height * 0.8), inplace=False)
        mesh = mesh + blade
    
    return apply_transforms(mesh, center, rotation, scale)

def create_lighthouse(base_radius=4, top_radius=2, height=20, 
                      center=(0, 0, 0), rotation=None, scale=None):
    """Create a lighthouse."""
    # Tapered tower (using cone frustum approximation)
    tower = pv.Cylinder(radius=base_radius, height=height * 0.8,
                        center=(center[0], center[1], center[2] + height*0.4),
                        direction=(0, 0, 1))
    # Top observation deck
    deck = pv.Cylinder(radius=top_radius * 1.5, height=height * 0.1,
                       center=(center[0], center[1], center[2] + height * 0.85),
                       direction=(0, 0, 1))
    # Light housing
    light = pv.Cylinder(radius=top_radius, height=height * 0.1,
                        center=(center[0], center[1], center[2] + height * 0.95),
                        direction=(0, 0, 1), resolution=16)
    # Dome top
    dome = pv.Sphere(radius=top_radius * 0.8,
                     center=(center[0], center[1], center[2] + height))
    
    mesh = tower + deck + light + dome
    return apply_transforms(mesh, center, rotation, scale)

# ============== NATURE ==============

def create_bush(radius=1.5, center=(0, 0, 0), rotation=None, scale=None):
    """Create a bush (cluster of spheres)."""
    main = pv.Sphere(radius=radius, center=(center[0], center[1], center[2] + radius*0.8))
    # Add smaller spheres around
    mesh = main
    for angle in range(0, 360, 60):
        rad = np.radians(angle)
        x = center[0] + radius * 0.6 * np.cos(rad)
        y = center[1] + radius * 0.6 * np.sin(rad)
        small = pv.Sphere(radius=radius * 0.6, center=(x, y, center[2] + radius*0.5))
        mesh = mesh + small
    return apply_transforms(mesh, center, rotation, scale)

def create_rock(width=2, height=1.5, depth=2, center=(0, 0, 0), 
                rotation=None, scale=None, seed=None):
    """Create an irregular rock shape."""
    if seed:
        np.random.seed(seed)
    # Start with ellipsoid
    mesh = pv.ParametricEllipsoid(width/2, depth/2, height/2)
    mesh = mesh.translate((center[0], center[1], center[2] + height/2), inplace=False)
    # Add noise for irregularity
    points = mesh.points.copy()
    noise = np.random.uniform(-0.15, 0.15, points.shape) * np.array([width, depth, height])
    points += noise
    mesh.points = points
    return apply_transforms(mesh, center, rotation, scale)

def create_mountain(base_radius=20, height=15, peaks=3, 
                    center=(0, 0, 0), rotation=None, scale=None):
    """Create a mountain with multiple peaks."""
    meshes = []
    # Main peak
    main = pv.Cone(radius=base_radius, height=height,
                   center=(center[0], center[1], center[2] + height/2),
                   direction=(0, 0, 1), resolution=20)
    meshes.append(main)
    
    # Secondary peaks
    for i in range(peaks - 1):
        angle = (i + 1) * 2 * np.pi / peaks
        dist = base_radius * 0.4
        x = center[0] + dist * np.cos(angle)
        y = center[1] + dist * np.sin(angle)
        h = height * (0.6 + 0.2 * np.random.random())
        r = base_radius * (0.4 + 0.2 * np.random.random())
        peak = pv.Cone(radius=r, height=h, center=(x, y, center[2] + h/2),
                       direction=(0, 0, 1), resolution=16)
        meshes.append(peak)
    
    mesh = meshes[0]
    for m in meshes[1:]:
        mesh = mesh + m
    return apply_transforms(mesh, center, rotation, scale)

def create_pond(radius=8, depth=0.5, center=(0, 0, 0), rotation=None, scale=None):
    """Create a pond (recessed disc)."""
    # Water surface (slightly below ground)
    water = pv.Disc(center=(center[0], center[1], center[2] - depth), 
                    normal=(0, 0, 1), inner=0, outer=radius, c_res=40)
    # Simple representation
    return apply_transforms(water, center, rotation, scale)

def create_river(length=40, width=5, center=(0, 0, 0), rotation=None, scale=None):
    """Create a river segment."""
    river = pv.Plane(center=(center[0], center[1], center[2] - 0.1),
                     direction=(0, 0, 1), i_size=length, j_size=width)
    return apply_transforms(river, center, rotation, scale)

def create_flower(stem_height=1, petal_radius=0.3, center=(0, 0, 0), 
                  rotation=None, scale=None):
    """Create a simple flower."""
    # Stem
    stem = pv.Cylinder(radius=0.05, height=stem_height,
                       center=(center[0], center[1], center[2] + stem_height/2),
                       direction=(0, 0, 1))
    # Center
    flower_center = pv.Sphere(radius=petal_radius * 0.4,
                               center=(center[0], center[1], center[2] + stem_height))
    # Petals
    mesh = stem + flower_center
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        x = center[0] + petal_radius * 0.7 * np.cos(rad)
        y = center[1] + petal_radius * 0.7 * np.sin(rad)
        petal = pv.Sphere(radius=petal_radius * 0.5, 
                          center=(x, y, center[2] + stem_height))
        mesh = mesh + petal
    
    return apply_transforms(mesh, center, rotation, scale)

# ============== FURNITURE & OBJECTS ==============

def create_table(width=4, depth=3, height=2.5, leg_width=0.2,
                 center=(0, 0, 0), rotation=None, scale=None):
    """Create a table."""
    # Tabletop
    top = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2] + height - 0.15, center[2] + height
    ))
    # Legs
    mesh = top
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        leg = pv.Box(bounds=(
            center[0] + dx*(width/2 - leg_width) - leg_width/2,
            center[0] + dx*(width/2 - leg_width) + leg_width/2,
            center[1] + dy*(depth/2 - leg_width) - leg_width/2,
            center[1] + dy*(depth/2 - leg_width) + leg_width/2,
            center[2], center[2] + height - 0.15
        ))
        mesh = mesh + leg
    
    return apply_transforms(mesh, center, rotation, scale)

def create_chair(width=1.5, depth=1.5, seat_height=1.5, back_height=2.5,
                 center=(0, 0, 0), rotation=None, scale=None):
    """Create a chair."""
    leg_w = 0.1
    # Seat
    seat = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2] + seat_height - 0.1, center[2] + seat_height
    ))
    # Back
    back = pv.Box(bounds=(
        center[0] - width/2, center[0] + width/2,
        center[1] + depth/2 - 0.1, center[1] + depth/2,
        center[2] + seat_height, center[2] + back_height
    ))
    mesh = seat + back
    # Legs
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        leg = pv.Box(bounds=(
            center[0] + dx*(width/2 - leg_w) - leg_w/2,
            center[0] + dx*(width/2 - leg_w) + leg_w/2,
            center[1] + dy*(depth/2 - leg_w) - leg_w/2,
            center[1] + dy*(depth/2 - leg_w) + leg_w/2,
            center[2], center[2] + seat_height - 0.1
        ))
        mesh = mesh + leg
    
    return apply_transforms(mesh, center, rotation, scale)

def create_bench(length=4, depth=1, height=1.5, center=(0, 0, 0), 
                 rotation=None, scale=None):
    """Create a park bench."""
    # Seat
    seat = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] - depth/2, center[1] + depth/2,
        center[2] + height - 0.15, center[2] + height
    ))
    # Back
    back = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] + depth/2 - 0.1, center[1] + depth/2,
        center[2] + height, center[2] + height + depth
    ))
    # Legs
    mesh = seat + back
    for dx in [-1, 1]:
        leg = pv.Box(bounds=(
            center[0] + dx * (length/2 - 0.15) - 0.1,
            center[0] + dx * (length/2 - 0.15) + 0.1,
            center[1] - depth/2, center[1] + depth/2,
            center[2], center[2] + height - 0.15
        ))
        mesh = mesh + leg
    
    return apply_transforms(mesh, center, rotation, scale)

def create_lamp_post(height=8, center=(0, 0, 0), rotation=None, scale=None):
    """Create a street lamp."""
    # Post
    post = pv.Cylinder(radius=0.15, height=height,
                       center=(center[0], center[1], center[2] + height/2),
                       direction=(0, 0, 1), resolution=12)
    # Lamp housing
    lamp = pv.Sphere(radius=0.6, center=(center[0], center[1], center[2] + height))
    # Arm
    arm = pv.Cylinder(radius=0.08, height=1,
                      center=(center[0] + 0.5, center[1], center[2] + height - 0.5),
                      direction=(1, 0, 0))
    
    mesh = post + lamp + arm
    return apply_transforms(mesh, center, rotation, scale)

def create_fountain(radius=4, height=3, center=(0, 0, 0), 
                    rotation=None, scale=None):
    """Create a fountain."""
    # Base pool
    pool = pv.Cylinder(radius=radius, height=0.5,
                       center=(center[0], center[1], center[2] + 0.25),
                       direction=(0, 0, 1))
    pool_inner = pv.Cylinder(radius=radius - 0.3, height=0.4,
                             center=(center[0], center[1], center[2] + 0.3),
                             direction=(0, 0, 1))
    # Center pedestal
    pedestal = pv.Cylinder(radius=0.6, height=height,
                           center=(center[0], center[1], center[2] + height/2),
                           direction=(0, 0, 1))
    # Top bowl
    bowl = pv.Cylinder(radius=1.2, height=0.3,
                       center=(center[0], center[1], center[2] + height),
                       direction=(0, 0, 1))
    
    mesh = pool + pedestal + bowl
    return apply_transforms(mesh, center, rotation, scale)

def create_statue(height=5, base_size=2, center=(0, 0, 0), 
                  rotation=None, scale=None):
    """Create a simple statue on pedestal."""
    # Pedestal
    pedestal = pv.Box(bounds=(
        center[0] - base_size/2, center[0] + base_size/2,
        center[1] - base_size/2, center[1] + base_size/2,
        center[2], center[2] + height * 0.3
    ))
    # Figure (simplified as cylinder + sphere)
    body = pv.Cylinder(radius=height * 0.1, height=height * 0.5,
                       center=(center[0], center[1], center[2] + height*0.3 + height*0.25),
                       direction=(0, 0, 1))
    head = pv.Sphere(radius=height * 0.12,
                     center=(center[0], center[1], center[2] + height*0.3 + height*0.5 + height*0.12))
    
    mesh = pedestal + body + head
    return apply_transforms(mesh, center, rotation, scale)

# ============== VEHICLES (simplified) ==============

def create_car(length=4, width=2, height=1.5, center=(0, 0, 0), 
               rotation=None, scale=None):
    """Create a simple car shape."""
    # Body
    body = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] - width/2, center[1] + width/2,
        center[2] + 0.4, center[2] + height * 0.6
    ))
    # Cabin
    cabin = pv.Box(bounds=(
        center[0] - length * 0.25, center[0] + length * 0.2,
        center[1] - width/2 + 0.1, center[1] + width/2 - 0.1,
        center[2] + height * 0.6, center[2] + height
    ))
    # Wheels
    mesh = body + cabin
    for dx, dy in [(-0.35, -1), (-0.35, 1), (0.35, -1), (0.35, 1)]:
        wheel = pv.Cylinder(radius=0.35, height=0.2,
                            center=(center[0] + dx * length, center[1] + dy * (width/2 + 0.1), center[2] + 0.35),
                            direction=(0, 1, 0), resolution=16)
        mesh = mesh + wheel
    
    return apply_transforms(mesh, center, rotation, scale)

def create_boat(length=6, width=2, height=1.5, center=(0, 0, 0), 
                rotation=None, scale=None):
    """Create a simple boat."""
    # Hull (simplified as box)
    hull = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] - width/2, center[1] + width/2,
        center[2], center[2] + height * 0.4
    ))
    # Cabin
    cabin = pv.Box(bounds=(
        center[0] - length * 0.1, center[0] + length * 0.2,
        center[1] - width * 0.3, center[1] + width * 0.3,
        center[2] + height * 0.4, center[2] + height
    ))
    
    mesh = hull + cabin
    return apply_transforms(mesh, center, rotation, scale)

def create_bridge(length=20, width=6, height=3, arch_count=3,
                  center=(0, 0, 0), rotation=None, scale=None):
    """Create a bridge with arches."""
    # Deck
    deck = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] - width/2, center[1] + width/2,
        center[2] + height - 0.5, center[2] + height
    ))
    # Railings
    left_rail = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] - width/2, center[1] - width/2 + 0.2,
        center[2] + height, center[2] + height + 1
    ))
    right_rail = pv.Box(bounds=(
        center[0] - length/2, center[0] + length/2,
        center[1] + width/2 - 0.2, center[1] + width/2,
        center[2] + height, center[2] + height + 1
    ))
    # Pillars
    mesh = deck + left_rail + right_rail
    pillar_spacing = length / (arch_count + 1)
    for i in range(arch_count + 2):
        x = center[0] - length/2 + i * pillar_spacing
        pillar = pv.Box(bounds=(
            x - 0.5, x + 0.5,
            center[1] - width/2, center[1] + width/2,
            center[2], center[2] + height - 0.5
        ))
        mesh = mesh + pillar
    
    return apply_transforms(mesh, center, rotation, scale)

# Shape registry
SHAPES = {
    # Basic primitives
    "box": create_box,
    "rounded_box": create_rounded_box,
    "cylinder": create_cylinder,
    "cone": create_cone,
    "sphere": create_sphere,
    "hemisphere": create_hemisphere,
    "ellipsoid": create_ellipsoid,
    "plane": create_plane,
    "terrain": create_terrain,
    "pyramid": create_pyramid,
    "prism": create_prism,
    "torus": create_torus,
    "disc": create_disc,
    "tube": create_tube,
    "arch": create_arch,
    "stairs": create_stairs,
    "wedge": create_wedge,
    # Compound shapes - Architecture
    "tree": create_tree,
    "wall": create_wall,
    "roof": create_roof,
    "fence": create_fence,
    "column": create_column,
    "house": create_house,
    "tower": create_tower,
    "castle": create_castle,
    "church": create_church,
    "skyscraper": create_skyscraper,
    "windmill": create_windmill,
    "lighthouse": create_lighthouse,
    # Compound shapes - Nature
    "bush": create_bush,
    "rock": create_rock,
    "mountain": create_mountain,
    "pond": create_pond,
    "river": create_river,
    "flower": create_flower,
    # Compound shapes - Objects
    "table": create_table,
    "chair": create_chair,
    "bench": create_bench,
    "lamp_post": create_lamp_post,
    "fountain": create_fountain,
    "statue": create_statue,
    # Compound shapes - Vehicles
    "car": create_car,
    "boat": create_boat,
    "bridge": create_bridge,
    # Arrays
    "array": create_array,
    "circular_array": create_circular_array,
}

# ============== SCENE BUILDER ==============

# Map custom colors to hex values (PyVista compatible)
COLOR_MAP = {
    "darkgreen": "#006400",
    "lightgreen": "#90EE90",
    "darkbrown": "#654321",
    "beige": "#F5F5DC",
    "skyblue": "#87CEEB",
    "navy": "#000080",
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "darkgray": "#404040",
    "lightgray": "#D3D3D3",
    "darkblue": "#00008B",
    "lightblue": "#ADD8E6",
    "darkred": "#8B0000",
    "ivory": "#FFFFF0",
    "tan": "#D2B48C",
    "olive": "#808000",
    "maroon": "#800000",
    "aqua": "#00FFFF",
    "coral": "#FF7F50",
    "salmon": "#FA8072",
    "khaki": "#F0E68C",
    "crimson": "#DC143C",
    "indigo": "#4B0082",
    "violet": "#EE82EE",
    "turquoise": "#40E0D0",
    "slate": "#708090",
    "charcoal": "#36454F",
    "sand": "#C2B280",
    "forest": "#228B22",
    "brick": "#CB4154",
    "stone": "#928E85",
    "wood": "#DEB887",
}

def get_color(color_name):
    """Convert color name to PyVista-compatible format."""
    if not color_name:
        return "gray"
    color_lower = color_name.lower().replace(" ", "")
    return COLOR_MAP.get(color_lower, color_name)

def build_scene(objects):
    """Build a scene from a list of object definitions."""
    meshes = []
    for obj in objects:
        shape_type = obj.get("type", "box")
        color = get_color(obj.get("color", "gray"))
        # Extract rotation if present, convert to dict format
        rotation = obj.get("rotation", None)
        if isinstance(rotation, list) and len(rotation) == 3:
            rotation = {"x": rotation[0], "y": rotation[1], "z": rotation[2]}
        
        # Get shape-specific params (exclude type, color)
        params = {k: v for k, v in obj.items() if k not in ["type", "color"]}
        
        if shape_type in SHAPES:
            try:
                mesh = SHAPES[shape_type](**params)
                meshes.append({"mesh": mesh, "color": color})
            except Exception as e:
                print(f"  Warning: Failed to create {shape_type}: {e}")
    return meshes

# ============== AI INTERFACE ==============

SYSTEM_PROMPT = """You are an expert 3D scene architect. Generate precise, detailed 3D scenes using compound shapes when possible.

COORDINATE SYSTEM: X(left-right), Y(front-back), Z(ground-sky). Ground is z=0.

COMMON PARAMS: center=[x,y,z], rotation={x,y,z}, scale=number or [sx,sy,sz]

═══════════════════════════════════════════════════════════
BASIC PRIMITIVES
═══════════════════════════════════════════════════════════
box: width, height, depth
cylinder: radius, height
cone: radius, height
sphere: radius
hemisphere: radius (dome)
ellipsoid: radius_x, radius_y, radius_z
pyramid: base, height, sides (3-8)
prism: radius, height, sides
torus: ring_radius, tube_radius
disc: radius, inner_radius
tube: radius, height, thickness
wedge: width, height, depth (ramp)
arch: width, height, depth, thickness
stairs: width, height, depth, steps
plane: width, depth
terrain: width, depth, roughness(1-5), max_height

═══════════════════════════════════════════════════════════
COMPOUND SHAPES - BUILDINGS (use these for faster, better results!)
═══════════════════════════════════════════════════════════
house: width, depth, height, roof_height, roof_style("gable"/"pyramid")
  → Complete house with walls, roof, door

tower: radius, height, roof_height, crenellations(bool)
  → Castle tower with battlements and conical roof

castle: width, depth, wall_height, tower_height
  → Full castle with 4 corner towers, walls, gate

church: width, depth, height, tower_height
  → Church with nave, gable roof, steeple

skyscraper: width, depth, height, floors
  → Modern building with floor lines

windmill: tower_height, blade_length
  → Windmill with tower, blades, roof

lighthouse: base_radius, top_radius, height
  → Lighthouse with tapered tower, light housing, dome

═══════════════════════════════════════════════════════════
COMPOUND SHAPES - NATURE
═══════════════════════════════════════════════════════════
tree: trunk_radius, trunk_height, crown_radius, crown_height, crown_type("cone"/"sphere"/"layered")
  → Complete tree with trunk and foliage

bush: radius
  → Cluster of spheres forming a bush

rock: width, height, depth, seed
  → Irregular rock shape

mountain: base_radius, height, peaks(1-5)
  → Mountain with multiple peaks

pond: radius, depth
  → Water surface

river: length, width
  → River segment

flower: stem_height, petal_radius
  → Simple flower with petals

═══════════════════════════════════════════════════════════
COMPOUND SHAPES - OBJECTS & FURNITURE
═══════════════════════════════════════════════════════════
table: width, depth, height, leg_width
  → Table with 4 legs

chair: width, depth, seat_height, back_height
  → Chair with back and legs

bench: length, depth, height
  → Park bench with backrest

lamp_post: height
  → Street lamp with post and light

fountain: radius, height
  → Fountain with pool, pedestal, bowl

statue: height, base_size
  → Statue on pedestal

column: radius, height, base_height, capital_height
  → Classical column

fence: length, height, post_spacing
  → Fence with posts and rails

wall: width, height, thickness
  → Simple wall

roof: width, depth, height, style("gable"/"pyramid"/"dome")
  → Standalone roof

═══════════════════════════════════════════════════════════
COMPOUND SHAPES - VEHICLES & STRUCTURES
═══════════════════════════════════════════════════════════
car: length, width, height
  → Simple car with body, cabin, wheels

boat: length, width, height
  → Boat with hull and cabin

bridge: length, width, height, arch_count
  → Bridge with deck, railings, pillars

═══════════════════════════════════════════════════════════
PATTERNS (for repetitive elements)
═══════════════════════════════════════════════════════════
array: base_type, count_x, count_y, count_z, spacing_x, spacing_y, spacing_z, + base params
circular_array: base_type, count, radius, + base params

═══════════════════════════════════════════════════════════
COLORS
═══════════════════════════════════════════════════════════
Basic: red, green, blue, yellow, brown, gray, white, black, orange, purple, pink, cyan, teal
Earth: darkgreen, forest, lightgreen, darkbrown, beige, tan, sand, wood, brick, stone
Water/Sky: skyblue, navy, lightblue, aqua, turquoise
Metal: gold, silver, slate, charcoal
Other: ivory, coral, salmon, khaki, crimson, violet

═══════════════════════════════════════════════════════════
EXAMPLE - "a village scene"
═══════════════════════════════════════════════════════════
{
  "objects": [
    {"type": "terrain", "width": 100, "depth": 100, "roughness": 2, "max_height": 3, "center": [0,0,0], "color": "darkgreen"},
    {"type": "house", "width": 10, "depth": 12, "height": 8, "roof_height": 4, "center": [0,0,3], "color": "beige"},
    {"type": "house", "width": 8, "depth": 10, "height": 6, "roof_height": 3, "center": [-20,10,3], "color": "white"},
    {"type": "church", "width": 12, "depth": 20, "height": 12, "tower_height": 25, "center": [25,0,3], "color": "stone"},
    {"type": "tree", "trunk_height": 6, "crown_height": 8, "crown_type": "layered", "center": [15,-15,3], "color": "forest"},
    {"type": "tree", "trunk_height": 5, "crown_height": 6, "crown_type": "sphere", "center": [-15,-10,3], "color": "darkgreen"},
    {"type": "fountain", "radius": 4, "height": 3, "center": [0,20,3], "color": "stone"},
    {"type": "bench", "length": 4, "center": [6,18,3], "color": "wood"},
    {"type": "lamp_post", "height": 6, "center": [-5,15,3], "color": "charcoal"},
    {"type": "fence", "length": 30, "height": 1.5, "center": [0,-25,3], "color": "wood"},
    {"type": "pond", "radius": 8, "center": [-30,-20,3], "color": "skyblue"}
  ]
}

TIPS:
- Use compound shapes (house, castle, tree, etc.) instead of building from primitives
- The AI generates better scenes with fewer, smarter objects
- Compound shapes have sensible defaults - only specify params you want to change
- Use terrain for natural ground with rolling hills
- Place objects slightly above terrain (z=2-5) when using terrain

Return ONLY valid JSON with "objects" array."""

def get_scene_from_ai(prompt):
    """Get scene definition from AI."""
    print("  Calling OpenAI API...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        timeout=30.0
    )
    print("  Parsing response...")
    return json.loads(response.choices[0].message.content)

# ============== MAIN ==============

if __name__ == "__main__":
    print("\n🏗️  AI 3D Scene Generator")
    print("=" * 50)
    print("Examples:")
    print("  - 'a simple house with a tree'")
    print("  - 'a landscape with hills and a lake'")
    print("  - 'a castle with towers'")
    print("  - 'a snowman'")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get prompt first
        prompt = input("Describe a scene (or 'quit'): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
            
        print(f"🔨 Generating: {prompt}...")
        try:
            scene_data = get_scene_from_ai(prompt)
            obj_count = len(scene_data.get("objects", []))
            print(f"✓ Created scene with {obj_count} objects")
            print("  Opening 3D viewer... (close window to continue)\n")
            
            # Build and display scene
            plotter = pv.Plotter()
            plotter.background_color = "lightblue"
            
            meshes = build_scene(scene_data.get("objects", []))
            for item in meshes:
                plotter.add_mesh(item["mesh"], color=item["color"], show_edges=False)
            
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.show()
            
        except Exception as e:
            print(f"✗ Error: {e}\n")
