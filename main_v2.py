"""
AI 3D Scene Generator - Three-Stage Architecture with Feedback Loop

Stage 1: LLM → Semantic Entities (high-level scene understanding)
Stage 2: Entity Agent → Primitive Objects (geometry generation)  
Stage 3: Vision Critic → Feedback (analyze rendered image, suggest fixes)

Feedback Loop:
  User Prompt → [Stage 1] → Entities → [Stage 2] → Render → Image
                    ↑                                         ↓
                    └──────── [Stage 3: Vision Critic] ←──────┘
"""

import pyvista as pv
import numpy as np
import json
import base64
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# ═══════════════════════════════════════════════════════════════════════════════
# PROCEDURAL TERRAIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_perlin_noise(shape, scale=0.1, octaves=4, persistence=0.5, seed=None):
    """Generate Perlin-like noise using multiple octaves of sine waves."""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = shape
    noise = np.zeros((h, w))
    
    for octave in range(octaves):
        freq = 2 ** octave
        amp = persistence ** octave
        
        # Random phase shifts for variety
        px, py = np.random.random() * 100, np.random.random() * 100
        
        x = np.linspace(0, freq * np.pi * scale, w) + px
        y = np.linspace(0, freq * np.pi * scale, h) + py
        xv, yv = np.meshgrid(x, y)
        
        noise += amp * (np.sin(xv) * np.cos(yv) + np.sin(xv * 0.7 + 1) * np.cos(yv * 1.3))
    
    # Normalize to 0-1 range
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def create_terrain_mesh(width=100, depth=100, resolution=80, center=(0, 0, 0),
                        height_scale=5, roughness=0.1, features=None, seed=42):
    """
    Create a natural terrain mesh with heightmap.
    
    features: list of dicts describing terrain features
        - {"type": "hill", "center": [x, y], "radius": r, "height": h}
        - {"type": "valley", "center": [x, y], "radius": r, "depth": d}
        - {"type": "plateau", "center": [x, y], "radius": r, "height": h}
        - {"type": "ridge", "start": [x1, y1], "end": [x2, y2], "width": w, "height": h}
    """
    # Create grid
    x = np.linspace(-width/2, width/2, resolution) + center[0]
    y = np.linspace(-depth/2, depth/2, resolution) + center[1]
    xv, yv = np.meshgrid(x, y)
    
    # Base noise terrain
    noise = create_perlin_noise((resolution, resolution), scale=roughness, seed=seed)
    zv = noise * height_scale + center[2]
    
    # Add terrain features
    if features:
        for feat in features:
            ftype = feat.get("type", "hill")
            
            if ftype == "hill":
                cx, cy = feat.get("center", [0, 0])
                r = feat.get("radius", 10)
                h = feat.get("height", 5)
                dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
                influence = np.maximum(0, 1 - dist / r)
                zv += h * influence ** 2  # Smooth falloff
                
            elif ftype == "mountain":
                cx, cy = feat.get("center", [0, 0])
                r = feat.get("radius", 20)
                h = feat.get("height", 15)
                # Sharper peak
                dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
                influence = np.maximum(0, 1 - dist / r)
                zv += h * influence ** 1.5
                # Add rocky noise near peak
                peak_noise = create_perlin_noise((resolution, resolution), scale=0.3, seed=seed+1)
                zv += peak_noise * h * 0.2 * influence
                
            elif ftype == "valley":
                cx, cy = feat.get("center", [0, 0])
                r = feat.get("radius", 10)
                d = feat.get("depth", 3)
                dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
                influence = np.maximum(0, 1 - dist / r)
                zv -= d * influence ** 2
                
            elif ftype == "plateau":
                cx, cy = feat.get("center", [0, 0])
                r = feat.get("radius", 15)
                h = feat.get("height", 3)
                dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
                # Flat top with steep edges
                influence = np.clip(1 - dist / r, 0, 1)
                influence = np.where(influence > 0.3, 1, influence / 0.3)
                zv += h * influence
                
            elif ftype == "ridge":
                start = np.array(feat.get("start", [-20, 0]))
                end = np.array(feat.get("end", [20, 0]))
                w = feat.get("width", 5)
                h = feat.get("height", 5)
                # Distance from line segment
                line = end - start
                line_len = np.linalg.norm(line)
                line_dir = line / line_len
                points = np.stack([xv, yv], axis=-1)
                to_points = points - start
                proj = np.clip(np.dot(to_points, line_dir), 0, line_len)
                closest = start + proj[..., np.newaxis] * line_dir
                dist = np.linalg.norm(points - closest, axis=-1)
                influence = np.maximum(0, 1 - dist / w)
                zv += h * influence ** 1.5
    
    # Create structured grid
    grid = pv.StructuredGrid(xv, yv, zv)
    return grid


def create_organic_tree(center=(0, 0, 0), height=10, trunk_radius=0.3, 
                        crown_radius=3, crown_style="natural", segments=8):
    """Create a more organic-looking tree with branching structure."""
    meshes = []
    
    # Trunk - slightly tapered cylinder with curve
    trunk_height = height * 0.4
    n_trunk_sections = 6
    trunk_points = []
    
    for i in range(n_trunk_sections + 1):
        t = i / n_trunk_sections
        z = center[2] + t * trunk_height
        # Slight curve in trunk
        x = center[0] + np.sin(t * np.pi * 0.3) * trunk_radius * 0.5
        y = center[1] + np.cos(t * np.pi * 0.2) * trunk_radius * 0.3
        r = trunk_radius * (1 - t * 0.4)  # Taper
        trunk_points.append((x, y, z, r))
    
    # Create trunk as connected cylinders
    for i in range(len(trunk_points) - 1):
        x1, y1, z1, r1 = trunk_points[i]
        x2, y2, z2, r2 = trunk_points[i + 1]
        seg_center = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
        seg_height = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        trunk_seg = pv.Cylinder(center=seg_center, direction=(x2-x1, y2-y1, z2-z1),
                                radius=(r1 + r2) / 2, height=seg_height, resolution=12)
        meshes.append({"mesh": trunk_seg, "color": "#5D4037"})  # Brown
    
    # Crown - multiple overlapping ellipsoids for natural look
    crown_base_z = center[2] + trunk_height
    
    if crown_style == "natural" or crown_style == "sphere":
        # Main crown
        np.random.seed(int(center[0] * 100 + center[1] * 10))
        n_blobs = 5 if crown_style == "natural" else 3
        
        for i in range(n_blobs):
            angle = 2 * np.pi * i / n_blobs + np.random.uniform(-0.3, 0.3)
            dist = crown_radius * np.random.uniform(0.2, 0.5)
            blob_x = center[0] + dist * np.cos(angle)
            blob_y = center[1] + dist * np.sin(angle)
            blob_z = crown_base_z + crown_radius * np.random.uniform(0.3, 0.8)
            blob_r = crown_radius * np.random.uniform(0.5, 0.8)
            
            blob = pv.Sphere(radius=blob_r, center=(blob_x, blob_y, blob_z), 
                            theta_resolution=16, phi_resolution=16)
            meshes.append({"mesh": blob, "color": "#2E7D32"})  # Green
        
        # Central crown mass
        main_crown = pv.Sphere(radius=crown_radius * 0.9, 
                               center=(center[0], center[1], crown_base_z + crown_radius * 0.5),
                               theta_resolution=20, phi_resolution=20)
        meshes.append({"mesh": main_crown, "color": "#388E3C"})
        
    elif crown_style == "cone" or crown_style == "layered":
        # Conifer style - stacked cones
        n_layers = 4
        for i in range(n_layers):
            t = i / n_layers
            layer_z = crown_base_z + t * height * 0.6
            layer_r = crown_radius * (1 - t * 0.6)
            layer_h = height * 0.2
            cone = pv.Cone(center=(center[0], center[1], layer_z + layer_h/2),
                          direction=(0, 0, 1), radius=layer_r, height=layer_h, resolution=16)
            shade = f"#{int(30 + t * 20):02X}{int(100 + t * 40):02X}{int(30 + t * 20):02X}"
            meshes.append({"mesh": cone, "color": "#1B5E20" if i % 2 == 0 else "#2E7D32"})
    
    return meshes


def create_organic_rock(center=(0, 0, 0), size=2, seed=None):
    """Create a natural-looking rock using deformed icosphere."""
    if seed is None:
        seed = int(abs(center[0] * 100 + center[1] * 10 + center[2]))
    np.random.seed(seed)
    
    # Create base icosphere
    rock = pv.Icosphere(radius=size, nsub=2)
    
    # Deform vertices for organic shape
    points = rock.points.copy()
    
    # Random scaling for asymmetry
    scale = np.array([
        1 + np.random.uniform(-0.3, 0.3),
        1 + np.random.uniform(-0.3, 0.3),
        0.6 + np.random.uniform(-0.2, 0.2)  # Flatter
    ])
    points *= scale
    
    # Add noise to each vertex
    for i in range(len(points)):
        noise = np.random.uniform(-0.2, 0.2, 3) * size
        points[i] += noise
    
    rock.points = points
    
    # Position
    rock = rock.translate((center[0], center[1], center[2] + size * 0.3), inplace=False)
    
    return rock


def create_water_surface(center=(0, 0, 0), radius=10, resolution=40, wave_height=0.1):
    """Create animated-looking water surface with subtle waves."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    r = np.linspace(0, radius, resolution // 2)
    R, Theta = np.meshgrid(r, theta)
    
    X = R * np.cos(Theta) + center[0]
    Y = R * np.sin(Theta) + center[1]
    
    # Subtle wave pattern
    Z = wave_height * np.sin(R * 0.5) * np.cos(Theta * 3) + center[2]
    
    # Smooth edges
    edge_fade = np.clip(1 - R / radius, 0, 1) ** 0.5
    Z = Z * edge_fade + center[2] * (1 - edge_fade)
    
    grid = pv.StructuredGrid(X, Y, Z)
    return grid.extract_surface()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: ENTITY DECOMPOSITION AGENT
# Converts semantic entities to primitive objects
# ═══════════════════════════════════════════════════════════════════════════════

class EntityAgent:
    """
    Local agent that decomposes semantic entities into primitive objects.
    This is deterministic - no LLM calls, just rules.
    """
    
    @staticmethod
    def decompose(entity: dict) -> list:
        """Convert a semantic entity to a list of primitive objects."""
        entity_type = entity.get("type", "").lower()
        
        # Get the decomposition function
        decomposer = getattr(EntityAgent, f"_decompose_{entity_type}", None)
        if decomposer:
            return decomposer(entity)
        else:
            # Fallback: treat as a primitive
            return [entity]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BUILDING DECOMPOSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _decompose_house(entity: dict) -> list:
        """Decompose house into: body, roof, door, windows."""
        center = entity.get("center", [0, 0, 0])
        width = entity.get("width", 10)
        depth = entity.get("depth", 10)
        height = entity.get("height", 8)
        roof_height = entity.get("roof_height", 4)
        style = entity.get("style", "simple")
        wall_color = entity.get("wall_color", entity.get("color", "beige"))
        roof_color = entity.get("roof_color", "brown")
        door_color = entity.get("door_color", "wood")
        window_color = entity.get("window_color", "skyblue")
        
        primitives = []
        
        # Main body
        primitives.append({
            "type": "box",
            "width": width, "height": height, "depth": depth,
            "center": center,
            "color": wall_color
        })
        
        # Roof
        if style == "modern":
            primitives.append({
                "type": "box",
                "width": width * 1.05, "height": roof_height * 0.3, "depth": depth * 1.05,
                "center": [center[0], center[1], center[2] + height],
                "color": roof_color
            })
        else:  # gable roof
            primitives.append({
                "type": "pyramid",
                "base": max(width, depth) * 1.15, "height": roof_height, "sides": 4,
                "center": [center[0], center[1], center[2] + height],
                "color": roof_color
            })
        
        # Door
        door_width = width * 0.2
        door_height = height * 0.45
        primitives.append({
            "type": "box",
            "width": door_width, "height": door_height, "depth": 0.3,
            "center": [center[0], center[1] - depth/2, center[2]],
            "color": door_color
        })
        
        # Windows (2 on front)
        window_size = min(width, height) * 0.15
        window_z = center[2] + height * 0.5
        for dx in [-1, 1]:
            primitives.append({
                "type": "box",
                "width": window_size, "height": window_size, "depth": 0.2,
                "center": [center[0] + dx * width * 0.3, center[1] - depth/2, window_z],
                "color": window_color
            })
        
        # Chimney (optional based on style)
        if style != "modern":
            primitives.append({
                "type": "box",
                "width": width * 0.12, "height": roof_height * 0.8, "depth": width * 0.12,
                "center": [center[0] + width * 0.25, center[1], center[2] + height + roof_height * 0.3],
                "color": "brick"
            })
        
        return primitives
    
    @staticmethod
    def _decompose_tree(entity: dict) -> list:
        """Decompose tree using organic procedural generation."""
        center = entity.get("center", [0, 0, 0])
        height = entity.get("height", 10)
        crown_style = entity.get("crown_style", "natural")  # natural, sphere, cone, layered
        
        # Return special organic_tree type for procedural rendering
        return [{
            "type": "organic_tree",
            "center": center,
            "height": height,
            "trunk_radius": height * 0.05,
            "crown_radius": height * 0.25,
            "crown_style": crown_style
        }]
    
    @staticmethod
    def _decompose_tower(entity: dict) -> list:
        """Decompose tower into: body, roof, battlements."""
        center = entity.get("center", [0, 0, 0])
        radius = entity.get("radius", 3)
        height = entity.get("height", 15)
        roof_height = entity.get("roof_height", 5)
        has_battlements = entity.get("battlements", True)
        body_color = entity.get("body_color", entity.get("color", "stone"))
        roof_color = entity.get("roof_color", "darkbrown")
        
        primitives = []
        
        # Main body
        primitives.append({
            "type": "cylinder",
            "radius": radius, "height": height,
            "center": center,
            "color": body_color
        })
        
        # Conical roof
        primitives.append({
            "type": "cone",
            "radius": radius * 1.2, "height": roof_height,
            "center": [center[0], center[1], center[2] + height],
            "color": roof_color
        })
        
        # Battlements
        if has_battlements:
            n_battlements = 8
            for i in range(n_battlements):
                angle = 2 * np.pi * i / n_battlements
                x = center[0] + radius * 0.85 * np.cos(angle)
                y = center[1] + radius * 0.85 * np.sin(angle)
                primitives.append({
                    "type": "box",
                    "width": 0.6, "height": 1.2, "depth": 0.6,
                    "center": [x, y, center[2] + height],
                    "color": body_color
                })
        
        return primitives
    
    @staticmethod
    def _decompose_castle(entity: dict) -> list:
        """Decompose castle into: towers, walls, gate."""
        center = entity.get("center", [0, 0, 0])
        width = entity.get("width", 30)
        depth = entity.get("depth", 30)
        wall_height = entity.get("wall_height", 10)
        tower_height = entity.get("tower_height", 18)
        wall_color = entity.get("wall_color", entity.get("color", "stone"))
        tower_radius = entity.get("tower_radius", 3)
        
        primitives = []
        
        # Four corner towers
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            tower_prims = EntityAgent._decompose_tower({
                "center": [center[0] + dx * width/2, center[1] + dy * depth/2, center[2]],
                "radius": tower_radius,
                "height": tower_height,
                "roof_height": 5,
                "color": wall_color
            })
            primitives.extend(tower_prims)
        
        # Walls
        wall_thickness = 2
        # Front and back walls
        for dy in [-depth/2, depth/2]:
            primitives.append({
                "type": "box",
                "width": width - tower_radius * 2,
                "height": wall_height,
                "depth": wall_thickness,
                "center": [center[0], center[1] + dy, center[2]],
                "color": wall_color
            })
        # Side walls
        for dx in [-width/2, width/2]:
            primitives.append({
                "type": "box",
                "width": wall_thickness,
                "height": wall_height,
                "depth": depth - tower_radius * 2,
                "center": [center[0] + dx, center[1], center[2]],
                "color": wall_color
            })
        
        # Gate (front wall opening represented by arch-like structure)
        primitives.append({
            "type": "box",
            "width": 4, "height": 6, "depth": wall_thickness + 0.5,
            "center": [center[0], center[1] - depth/2, center[2]],
            "color": "darkbrown"
        })
        
        return primitives
    
    @staticmethod
    def _decompose_church(entity: dict) -> list:
        """Decompose church into: nave, roof, tower, steeple."""
        center = entity.get("center", [0, 0, 0])
        width = entity.get("width", 12)
        depth = entity.get("depth", 20)
        height = entity.get("height", 12)
        tower_height = entity.get("tower_height", 25)
        body_color = entity.get("body_color", entity.get("color", "stone"))
        roof_color = entity.get("roof_color", "darkbrown")
        
        primitives = []
        
        # Nave (main building)
        primitives.append({
            "type": "box",
            "width": width, "height": height, "depth": depth,
            "center": center,
            "color": body_color
        })
        
        # Gable roof
        primitives.append({
            "type": "pyramid",
            "base": max(width, depth) * 1.1, "height": height * 0.4, "sides": 4,
            "center": [center[0], center[1], center[2] + height],
            "color": roof_color
        })
        
        # Bell tower
        tower_w = width * 0.35
        primitives.append({
            "type": "box",
            "width": tower_w, "height": tower_height, "depth": tower_w,
            "center": [center[0], center[1] + depth/2 - tower_w/2, center[2]],
            "color": body_color
        })
        
        # Steeple
        primitives.append({
            "type": "cone",
            "radius": tower_w * 0.6, "height": tower_height * 0.3,
            "center": [center[0], center[1] + depth/2 - tower_w/2, center[2] + tower_height],
            "color": roof_color
        })
        
        # Door
        primitives.append({
            "type": "box",
            "width": width * 0.25, "height": height * 0.4, "depth": 0.3,
            "center": [center[0], center[1] - depth/2, center[2]],
            "color": "darkbrown"
        })
        
        return primitives

    @staticmethod
    def _decompose_fountain(entity: dict) -> list:
        """Decompose fountain into: pool, pedestal, bowl, water."""
        center = entity.get("center", [0, 0, 0])
        radius = entity.get("radius", 4)
        height = entity.get("height", 3)
        material_color = entity.get("color", "stone")
        water_color = entity.get("water_color", "skyblue")
        
        primitives = []
        
        # Base pool
        primitives.append({
            "type": "cylinder",
            "radius": radius, "height": 0.5,
            "center": center,
            "color": material_color
        })
        
        # Pool water
        primitives.append({
            "type": "disc",
            "radius": radius - 0.3,
            "center": [center[0], center[1], center[2] + 0.4],
            "color": water_color
        })
        
        # Center pedestal
        primitives.append({
            "type": "cylinder",
            "radius": 0.5, "height": height,
            "center": center,
            "color": material_color
        })
        
        # Top bowl
        primitives.append({
            "type": "cylinder",
            "radius": 1.2, "height": 0.4,
            "center": [center[0], center[1], center[2] + height],
            "color": material_color
        })
        
        return primitives
    
    @staticmethod
    def _decompose_bench(entity: dict) -> list:
        """Decompose bench into: seat, back, legs."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 4)
        depth = entity.get("depth", 1)
        height = entity.get("height", 1.5)
        color = entity.get("color", "wood")
        
        primitives = []
        
        # Seat
        primitives.append({
            "type": "box",
            "width": length, "height": 0.15, "depth": depth,
            "center": [center[0], center[1], center[2] + height - 0.15],
            "color": color
        })
        
        # Back
        primitives.append({
            "type": "box",
            "width": length, "height": depth, "depth": 0.1,
            "center": [center[0], center[1] + depth/2, center[2] + height],
            "color": color
        })
        
        # Legs
        for dx in [-1, 1]:
            primitives.append({
                "type": "box",
                "width": 0.15, "height": height - 0.15, "depth": depth,
                "center": [center[0] + dx * (length/2 - 0.15), center[1], center[2]],
                "color": color
            })
        
        return primitives
    
    @staticmethod
    def _decompose_lamp_post(entity: dict) -> list:
        """Decompose lamp post into: pole, light."""
        center = entity.get("center", [0, 0, 0])
        height = entity.get("height", 6)
        pole_color = entity.get("pole_color", entity.get("color", "charcoal"))
        light_color = entity.get("light_color", "gold")
        
        primitives = []
        
        # Pole
        primitives.append({
            "type": "cylinder",
            "radius": 0.12, "height": height,
            "center": center,
            "color": pole_color
        })
        
        # Light globe
        primitives.append({
            "type": "sphere",
            "radius": 0.4,
            "center": [center[0], center[1], center[2] + height + 0.2],
            "color": light_color
        })
        
        return primitives
    
    @staticmethod
    def _decompose_bridge(entity: dict) -> list:
        """Decompose bridge into: deck, railings, pillars."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 20)
        width = entity.get("width", 6)
        height = entity.get("height", 3)
        pillars = entity.get("pillars", 3)
        deck_color = entity.get("deck_color", entity.get("color", "stone"))
        railing_color = entity.get("railing_color", "charcoal")
        
        primitives = []
        
        # Deck
        primitives.append({
            "type": "box",
            "width": length, "height": 0.5, "depth": width,
            "center": [center[0], center[1], center[2] + height],
            "color": deck_color
        })
        
        # Railings
        for dy in [-width/2, width/2]:
            primitives.append({
                "type": "box",
                "width": length, "height": 1, "depth": 0.15,
                "center": [center[0], center[1] + dy, center[2] + height + 0.5],
                "color": railing_color
            })
        
        # Pillars
        pillar_spacing = length / (pillars + 1)
        for i in range(pillars):
            x = center[0] - length/2 + (i + 1) * pillar_spacing
            primitives.append({
                "type": "box",
                "width": 1, "height": height, "depth": width,
                "center": [x, center[1], center[2]],
                "color": deck_color
            })
        
        return primitives
    
    @staticmethod
    def _decompose_car(entity: dict) -> list:
        """Decompose car into: body, cabin, wheels."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 4)
        width = entity.get("width", 2)
        body_color = entity.get("body_color", entity.get("color", "red"))
        
        primitives = []
        
        # Body
        primitives.append({
            "type": "box",
            "width": length, "height": 0.8, "depth": width,
            "center": [center[0], center[1], center[2] + 0.5],
            "color": body_color
        })
        
        # Cabin
        primitives.append({
            "type": "box",
            "width": length * 0.5, "height": 0.6, "depth": width * 0.9,
            "center": [center[0] - length * 0.05, center[1], center[2] + 1.3],
            "color": body_color
        })
        
        # Wheels
        for dx, dy in [(-0.35, -1), (-0.35, 1), (0.35, -1), (0.35, 1)]:
            primitives.append({
                "type": "cylinder",
                "radius": 0.35, "height": 0.2,
                "center": [center[0] + dx * length, center[1] + dy * (width/2 + 0.1), center[2] + 0.35],
                "rotation": {"x": 90},
                "color": "charcoal"
            })
        
        return primitives
    
    @staticmethod
    def _decompose_boat(entity: dict) -> list:
        """Decompose boat into: hull, cabin."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 6)
        width = entity.get("width", 2)
        hull_color = entity.get("hull_color", entity.get("color", "white"))
        cabin_color = entity.get("cabin_color", "white")
        
        primitives = []
        
        # Hull
        primitives.append({
            "type": "box",
            "width": length, "height": 0.6, "depth": width,
            "center": center,
            "color": hull_color
        })
        
        # Cabin
        primitives.append({
            "type": "box",
            "width": length * 0.3, "height": 0.8, "depth": width * 0.6,
            "center": [center[0], center[1], center[2] + 0.7],
            "color": cabin_color
        })
        
        return primitives
    
    @staticmethod
    def _decompose_mountain(entity: dict) -> list:
        """Decompose mountain using terrain heightmap with features."""
        center = entity.get("center", [0, 0, 0])
        base_radius = entity.get("base_radius", 20)
        height = entity.get("height", 15)
        peaks = entity.get("peaks", 3)
        color = entity.get("color", "gray")
        snow_color = entity.get("snow_color", "white")
        
        # Build terrain features
        features = [{"type": "mountain", "center": [center[0], center[1]], 
                     "radius": base_radius, "height": height}]
        
        # Secondary peaks
        np.random.seed(42)
        for i in range(peaks - 1):
            angle = (i + 1) * 2 * np.pi / peaks + np.random.uniform(-0.3, 0.3)
            dist = base_radius * 0.5
            h = height * (0.4 + 0.3 * np.random.random())
            r = base_radius * (0.3 + 0.2 * np.random.random())
            features.append({
                "type": "hill",
                "center": [center[0] + dist * np.cos(angle), center[1] + dist * np.sin(angle)],
                "radius": r, "height": h
            })
        
        return [{
            "type": "terrain",
            "width": base_radius * 2.5,
            "depth": base_radius * 2.5,
            "center": center,
            "features": features,
            "color": color,
            "height_scale": 2,
            "roughness": 0.15
        }]
        
        return primitives
    
    @staticmethod
    def _decompose_pond(entity: dict) -> list:
        """Decompose pond using organic water surface."""
        center = entity.get("center", [0, 0, 0])
        radius = entity.get("radius", 8)
        color = entity.get("color", "skyblue")
        
        return [{
            "type": "water_surface",
            "radius": radius,
            "center": center,
            "color": color
        }]
    
    @staticmethod
    def _decompose_rock(entity: dict) -> list:
        """Decompose rock using organic deformed mesh."""
        center = entity.get("center", [0, 0, 0])
        size = entity.get("size", 2)
        color = entity.get("color", "gray")
        
        return [{
            "type": "organic_rock",
            "size": size,
            "center": center,
            "color": color
        }]
    
    @staticmethod
    def _decompose_bush(entity: dict) -> list:
        """Decompose bush into: cluster of spheres."""
        center = entity.get("center", [0, 0, 0])
        radius = entity.get("radius", 1.5)
        color = entity.get("color", "darkgreen")
        
        primitives = []
        
        # Main sphere
        primitives.append({
            "type": "sphere",
            "radius": radius,
            "center": [center[0], center[1], center[2] + radius * 0.7],
            "color": color
        })
        
        # Surrounding smaller spheres
        for angle in range(0, 360, 72):
            rad = np.radians(angle)
            primitives.append({
                "type": "sphere",
                "radius": radius * 0.5,
                "center": [
                    center[0] + radius * 0.5 * np.cos(rad),
                    center[1] + radius * 0.5 * np.sin(rad),
                    center[2] + radius * 0.4
                ],
                "color": color
            })
        
        return primitives
    
    @staticmethod
    def _decompose_lighthouse(entity: dict) -> list:
        """Decompose lighthouse into: tower, light room, dome, stripes."""
        center = entity.get("center", [0, 0, 0])
        height = entity.get("height", 20)
        base_radius = entity.get("base_radius", 3)
        stripes = entity.get("stripes", True)
        color1 = entity.get("color", "white")
        color2 = entity.get("stripe_color", "red")
        light_color = entity.get("light_color", "gold")
        
        primitives = []
        top_radius = base_radius * 0.6
        
        # Main tower (tapered cylinder approximation with stacked cylinders)
        n_sections = 5 if stripes else 1
        section_height = height / n_sections
        for i in range(n_sections):
            r = base_radius - (base_radius - top_radius) * (i + 0.5) / n_sections
            z = center[2] + i * section_height
            primitives.append({
                "type": "cylinder",
                "radius": r, "height": section_height,
                "center": [center[0], center[1], z],
                "color": color1 if i % 2 == 0 else color2
            })
        
        # Light room
        room_height = height * 0.15
        primitives.append({
            "type": "cylinder",
            "radius": top_radius * 1.2, "height": room_height,
            "center": [center[0], center[1], center[2] + height],
            "color": "charcoal"
        })
        
        # Glass/light
        primitives.append({
            "type": "cylinder",
            "radius": top_radius * 0.8, "height": room_height * 0.8,
            "center": [center[0], center[1], center[2] + height],
            "color": light_color
        })
        
        # Dome
        primitives.append({
            "type": "cone",
            "radius": top_radius * 1.3, "height": height * 0.1,
            "center": [center[0], center[1], center[2] + height + room_height],
            "color": "charcoal"
        })
        
        return primitives
    
    @staticmethod
    def _decompose_island(entity: dict) -> list:
        """Decompose island into: land mass."""
        center = entity.get("center", [0, 0, 0])
        radius = entity.get("radius", 10)
        height = entity.get("height", 2)
        color = entity.get("color", "sand")
        
        primitives = []
        
        # Main land mass (flattened cone)
        primitives.append({
            "type": "cone",
            "radius": radius, "height": height,
            "center": [center[0], center[1], center[2] - height * 0.5],
            "color": color
        })
        
        # Top surface
        primitives.append({
            "type": "disc",
            "radius": radius * 0.6,
            "center": [center[0], center[1], center[2]],
            "color": "darkgreen"
        })
        
        return primitives
    
    @staticmethod  
    def _decompose_water(entity: dict) -> list:
        """Decompose water into: water surface plane."""
        center = entity.get("center", [0, 0, 0])
        width = entity.get("width", 100)
        depth = entity.get("depth", 100)
        color = entity.get("color", "ocean")
        
        return [{
            "type": "plane",
            "width": width, "depth": depth,
            "center": [center[0], center[1], center[2] - 0.1],
            "color": color
        }]
    
    @staticmethod
    def _decompose_road(entity: dict) -> list:
        """Decompose road into: surface plane."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 20)
        width = entity.get("width", 4)
        color = entity.get("color", "charcoal")
        
        return [{
            "type": "plane",
            "width": length, "depth": width,
            "center": center,
            "color": color
        }]
    
    @staticmethod
    def _decompose_path(entity: dict) -> list:
        """Decompose path into: surface plane."""
        center = entity.get("center", [0, 0, 0])
        length = entity.get("length", 15)
        width = entity.get("width", 2)
        color = entity.get("color", "sand")
        
        return [{
            "type": "plane",
            "width": length, "depth": width,
            "center": center,
            "color": color
        }]


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE RENDERER
# Creates PyVista meshes from primitive objects
# ═══════════════════════════════════════════════════════════════════════════════

def apply_transforms(mesh, center, rotation=None, scale=None):
    """Apply rotation and scale to a mesh."""
    if scale:
        if isinstance(scale, (int, float)):
            mesh = mesh.scale([scale, scale, scale], inplace=False)
        elif isinstance(scale, (list, tuple)):
            mesh = mesh.scale(scale, inplace=False)
    if rotation:
        rx = rotation.get('x', 0) if isinstance(rotation, dict) else 0
        ry = rotation.get('y', 0) if isinstance(rotation, dict) else 0
        rz = rotation.get('z', 0) if isinstance(rotation, dict) else 0
        if rx: mesh = mesh.rotate_x(rx, point=center, inplace=False)
        if ry: mesh = mesh.rotate_y(ry, point=center, inplace=False)
        if rz: mesh = mesh.rotate_z(rz, point=center, inplace=False)
    return mesh


def create_primitive(obj: dict):
    """Create a PyVista mesh from a primitive object definition.
    
    Returns either:
    - A single mesh (for standard primitives)
    - A list of {"mesh": mesh, "color": color} dicts (for organic/complex types)
    """
    ptype = obj.get("type", "box")
    center = tuple(obj.get("center", [0, 0, 0]))
    rotation = obj.get("rotation")
    scale = obj.get("scale")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ORGANIC/PROCEDURAL TYPES - return list of colored meshes
    # ═══════════════════════════════════════════════════════════════════════════
    
    if ptype == "organic_tree":
        height = obj.get("height", 10)
        trunk_radius = obj.get("trunk_radius", height * 0.05)
        crown_radius = obj.get("crown_radius", height * 0.25)
        crown_style = obj.get("crown_style", "natural")
        return create_organic_tree(center, height, trunk_radius, crown_radius, crown_style)
    
    elif ptype == "organic_rock":
        size = obj.get("size", 2)
        rock_mesh = create_organic_rock(center, size)
        return rock_mesh  # Single mesh, will be wrapped by build_scene
    
    elif ptype == "water_surface":
        radius = obj.get("radius", 10)
        return create_water_surface(center, radius)
    
    elif ptype == "terrain":
        width = obj.get("width", 100)
        depth = obj.get("depth", 100)
        features = obj.get("features", [])
        height_scale = obj.get("height_scale", 5)
        roughness = obj.get("roughness", 0.1)
        return create_terrain_mesh(width, depth, resolution=60, center=center,
                                   height_scale=height_scale, roughness=roughness,
                                   features=features)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD PRIMITIVES - return single mesh
    # ═══════════════════════════════════════════════════════════════════════════
    
    if ptype == "box":
        w, h, d = obj.get("width", 1), obj.get("height", 1), obj.get("depth", 1)
        mesh = pv.Box(bounds=(
            center[0] - w/2, center[0] + w/2,
            center[1] - d/2, center[1] + d/2,
            center[2], center[2] + h
        ))
    elif ptype == "cylinder":
        r, h = obj.get("radius", 1), obj.get("height", 1)
        mesh = pv.Cylinder(radius=r, height=h, 
                           center=(center[0], center[1], center[2] + h/2),
                           direction=(0, 0, 1), resolution=32)
    elif ptype == "cone":
        r, h = obj.get("radius", 1), obj.get("height", 1)
        mesh = pv.Cone(radius=r, height=h,
                       center=(center[0], center[1], center[2] + h/2),
                       direction=(0, 0, 1), resolution=32)
    elif ptype == "sphere":
        r = obj.get("radius", 1)
        mesh = pv.Sphere(radius=r, center=(center[0], center[1], center[2] + r))
    elif ptype == "ellipsoid":
        rx = obj.get("radius_x", 1)
        ry = obj.get("radius_y", 1)
        rz = obj.get("radius_z", 1)
        mesh = pv.ParametricEllipsoid(rx, ry, rz)
        mesh = mesh.translate((center[0], center[1], center[2] + rz), inplace=False)
    elif ptype == "pyramid":
        base, h = obj.get("base", 1), obj.get("height", 1)
        sides = obj.get("sides", 4)
        half = base / 2
        angles = np.linspace(0, 2*np.pi, sides+1)[:-1] + np.pi/sides
        points = [[center[0] + half*np.cos(a), center[1] + half*np.sin(a), center[2]] for a in angles]
        points.append([center[0], center[1], center[2] + h])
        points = np.array(points)
        faces = [[sides] + list(range(sides))]
        for i in range(sides):
            faces.append([3, i, (i+1) % sides, sides])
        mesh = pv.PolyData(points, np.hstack(faces))
    elif ptype == "disc":
        r = obj.get("radius", 1)
        inner = obj.get("inner_radius", 0)
        mesh = pv.Disc(center=center, normal=(0, 0, 1), inner=inner, outer=r, c_res=40)
    elif ptype == "plane":
        w, d = obj.get("width", 10), obj.get("depth", 10)
        mesh = pv.Plane(center=center, direction=(0, 0, 1), i_size=w, j_size=d)
    elif ptype == "torus":
        ring_r = obj.get("ring_radius", 2)
        tube_r = obj.get("tube_radius", 0.5)
        mesh = pv.ParametricTorus(ringradius=ring_r, crosssectionradius=tube_r)
        mesh = mesh.translate((center[0], center[1], center[2] + tube_r), inplace=False)
    else:
        # Fallback: small sphere
        mesh = pv.Sphere(radius=0.5, center=center)
    
    return apply_transforms(mesh, center, rotation, scale)


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_MAP = {
    # Greens
    "darkgreen": "#006400", "lightgreen": "#90EE90", "forest": "#228B22",
    "olive": "#808000", "seagreen": "#2E8B57", "mint": "#98FB98",
    "grass": "#7CFC00", "pine": "#01796F", "emerald": "#50C878",
    # Browns
    "darkbrown": "#654321", "brown": "#8B4513", "wood": "#DEB887", "wooden": "#DEB887",
    "beige": "#F5F5DC", "tan": "#D2B48C", "sand": "#C2B280", "sandy": "#C2B280",
    "chocolate": "#D2691E", "coffee": "#6F4E37", "lightwood": "#FAEBD7", "lightbrown": "#C4A484",
    "oak": "#806517", "walnut": "#773F1A", "mahogany": "#C04000",
    # Blues
    "skyblue": "#87CEEB", "navy": "#000080", "lightblue": "#ADD8E6",
    "turquoise": "#40E0D0", "aqua": "#00FFFF", "ocean": "#006994",
    "bluegray": "#6699CC", "steelblue": "#4682B4", "deepblue": "#00008B",
    "water": "#1CA3EC", "sea": "#006994", "azure": "#007FFF",
    # Grays
    "stone": "#928E85", "charcoal": "#36454F", "slate": "#708090",
    "darkgray": "#A9A9A9", "lightgray": "#D3D3D3", "ash": "#B2BEB5",
    "concrete": "#95A5A6", "cement": "#95A5A6", "marble": "#F0EAD6", "granite": "#676767",
    # Metals
    "gold": "#FFD700", "silver": "#C0C0C0", "bronze": "#CD7F32",
    "copper": "#B87333", "iron": "#48494B", "brass": "#B5A642",
    "steel": "#71797E", "metallic": "#AAA9AD",
    # Warm colors
    "brick": "#CB4154", "coral": "#FF7F50", "salmon": "#FA8072",
    "crimson": "#DC143C", "rust": "#B7410E", "terracotta": "#E2725B",
    "clay": "#B66A50", "amber": "#FFBF00",
    # Cool colors
    "violet": "#EE82EE", "lavender": "#E6E6FA", "plum": "#DDA0DD",
    # Neutrals
    "ivory": "#FFFFF0", "cream": "#FFFDD0", "offwhite": "#FAF9F6",
    "eggshell": "#F0EAD6", "pearl": "#EAE0C8", "bone": "#E3DAC9",
    # Misc
    "maroon": "#800000", "teal": "#008080", "indigo": "#4B0082",
    "cobblestone": "#433E37", "thatch": "#B8860B", "moss": "#8A9A5B",
    "snow": "#FFFAFA", "ice": "#A5F2F3", "iceblue": "#A5F2F3", "straw": "#E4D96F",
    "darkred": "#8B0000", "firebrick": "#B22222", "burgundy": "#800020",
    "lightyellow": "#FFFFE0", "paleyellow": "#FFFACD", "lemon": "#FFF44F",
}

# Standard CSS/matplotlib color names that PyVista accepts
VALID_COLORS = {
    'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 
    'magenta', 'white', 'black', 'gray', 'grey', 'brown', 'lime', 'olive',
    'navy', 'teal', 'aqua', 'maroon', 'silver', 'fuchsia'
}

def get_color(name):
    if not name:
        return "gray"
    
    normalized = name.lower().replace(" ", "").replace("-", "")
    
    # Check our custom color map first
    if normalized in COLOR_MAP:
        return COLOR_MAP[normalized]
    
    # Check if it's a standard color PyVista accepts
    if normalized in VALID_COLORS:
        return normalized
    
    # Fallback: return gray for unknown colors (prevents crashes)
    print(f"  Note: Unknown color '{name}' → using gray")
    return "gray"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: LLM SCENE UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_PROMPT = """You are a semantic scene planner. Convert natural language into structured scene entities.

YOUR JOB: Output semantic ENTITIES (not primitives). The system will decompose them into 3D shapes.

COORDINATE SYSTEM: X(left-right), Y(front-back), Z(up). Ground is z=0.

═══════════════════════════════════════════════════════════
AVAILABLE ENTITIES
═══════════════════════════════════════════════════════════

BUILDINGS:
- house: width, depth, height, roof_height, style("simple"/"modern"), wall_color, roof_color
- tower: radius, height, roof_height, battlements(bool), body_color, roof_color
- castle: width, depth, wall_height, tower_height, wall_color
- church: width, depth, height, tower_height, body_color, roof_color
- lighthouse: height, base_radius, stripes(bool), color, stripe_color, light_color

NATURE:
- tree: height, trunk_ratio(0.2-0.5), crown_style("cone"/"sphere"/"layered"), foliage_color, trunk_color
- bush: radius, color
- rock: size, color
- mountain: base_radius, height, peaks(1-5), color, snow_color
- pond: radius, color
- water: width, depth, color
- island: radius, height, color

SURFACES:
- road: length, width, color
- path: length, width, color

OBJECTS:
- fountain: radius, height, color, water_color
- bench: length, depth, height, color
- lamp_post: height, pole_color, light_color
- bridge: length, width, height, pillars, deck_color, railing_color

VEHICLES:
- car: length, width, body_color
- boat: length, width, hull_color, cabin_color

PRIMITIVES (for custom shapes not covered above):
- box: width, height, depth, center, color
- cylinder: radius, height, center, color
- cone: radius, height, center, color
- sphere: radius, center, color
- disc: radius, center, color
- plane: width, depth, center, color
- pyramid: base, height, sides, center, color

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════
{
  "scene_description": "Brief description of the scene",
  "ground": {"type": "plane", "width": 100, "depth": 100, "color": "green"},
  "entities": [
    {"type": "entity_type", "param1": value, "center": [x, y, z], ...},
    ...
  ]
}

Use appropriate positioning (center=[x, y, z]) to arrange entities spatially.
Return ONLY valid JSON."""


def get_entities_from_ai(prompt: str) -> dict:
    """Stage 1: Get semantic entities from LLM."""
    print("  [Stage 1] Calling LLM for semantic entities...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ENTITY_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        timeout=30.0
    )
    return json.loads(response.choices[0].message.content)


def process_scene(scene_data: dict) -> list:
    """Stage 2: Convert entities to primitives using EntityAgent."""
    print("  [Stage 2] Decomposing entities into primitives...")
    
    all_primitives = []
    
    # Process ground
    ground = scene_data.get("ground")
    if ground:
        all_primitives.append(ground)
    
    # Process each entity
    entities = scene_data.get("entities", [])
    for entity in entities:
        primitives = EntityAgent.decompose(entity)
        all_primitives.extend(primitives)
    
    return all_primitives


def build_scene(primitives: list):
    """Convert primitives to PyVista meshes for rendering.
    
    Handles:
    - Standard primitives → single mesh with color from prim
    - Organic types → list of meshes (already have colors) or single mesh
    """
    meshes = []
    for prim in primitives:
        try:
            result = create_primitive(prim)
            prim_color = get_color(prim.get("color", "gray"))
            
            # Check what type of result we got
            if isinstance(result, list):
                # List of {"mesh": mesh, "color": color} dicts (e.g., organic_tree)
                for item in result:
                    if isinstance(item, dict) and "mesh" in item:
                        meshes.append(item)
                    else:
                        # Just a mesh in a list
                        meshes.append({"mesh": item, "color": prim_color})
            elif hasattr(result, 'points'):
                # Single PyVista mesh
                meshes.append({"mesh": result, "color": prim_color})
            else:
                print(f"  Warning: Unknown result type for {prim.get('type')}")
                
        except Exception as e:
            print(f"  Warning: Failed to create {prim.get('type')}: {e}")
    return meshes


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: VISION CRITIC AGENT
# Analyzes rendered images and provides feedback for refinement
# ═══════════════════════════════════════════════════════════════════════════════

CRITIC_PROMPT = """You are a 3D scene critic. Analyze this rendered scene image and identify problems.

ORIGINAL USER REQUEST: {user_prompt}

CURRENT SCENE DATA:
{scene_json}

Analyze the image and identify:
1. MISSING ELEMENTS - Objects mentioned in the request but not visible
2. POSITIONING ISSUES - Objects that overlap, float, or are placed incorrectly
3. SCALE PROBLEMS - Objects that are too big/small relative to others
4. COLOR ISSUES - Colors that don't match the request or look wrong
5. COMPOSITION - Overall layout problems (too sparse, too crowded, etc.)

For each issue, provide a SPECIFIC FIX as a JSON patch.

OUTPUT FORMAT (JSON):
{{
  "analysis": "Brief description of what you see in the image",
  "score": 1-10,
  "issues": [
    {{
      "type": "missing|position|scale|color|composition",
      "description": "What's wrong",
      "entity_index": 0,
      "fix": {{
        "center": [x, y, z],
        "scale": 1.5,
        "color": "red"
      }}
    }}
  ],
  "should_refine": true,
  "refinement_instructions": "Natural language instructions for the scene generator"
}}

Be specific and actionable. If the scene looks good (score >= 8), set should_refine to false."""


def render_to_image(meshes: list, camera_position='iso') -> str:
    """Render scene to a temporary image file and return base64 encoded string."""
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
    plotter.background_color = "lightblue"
    
    for item in meshes:
        plotter.add_mesh(item["mesh"], color=item["color"], show_edges=False)
    
    plotter.camera_position = camera_position
    plotter.camera.zoom(0.8)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    plotter.screenshot(temp_path)
    plotter.close()
    
    # Read and encode as base64
    with open(temp_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Clean up
    os.unlink(temp_path)
    
    return image_data


def get_critic_feedback(image_base64: str, user_prompt: str, scene_data: dict) -> dict:
    """Stage 3: Get feedback from vision critic on the rendered scene."""
    print("  [Stage 3] Vision Critic analyzing scene...")
    
    prompt = CRITIC_PROMPT.format(
        user_prompt=user_prompt,
        scene_json=json.dumps(scene_data, indent=2)[:2000]  # Truncate if too long
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        timeout=60.0
    )
    
    return json.loads(response.choices[0].message.content)


def apply_fixes(scene_data: dict, feedback: dict) -> dict:
    """Apply critic's fixes to the scene data."""
    entities = scene_data.get("entities", [])
    
    for issue in feedback.get("issues", []):
        fix = issue.get("fix", {})
        entity_idx = issue.get("entity_index", -1)
        
        if entity_idx == -1 and "type" in fix:
            # Add new entity
            entities.append(fix)
            print(f"    + Added new {fix.get('type')}")
        elif 0 <= entity_idx < len(entities):
            # Update existing entity
            for key, value in fix.items():
                entities[entity_idx][key] = value
            print(f"    ~ Updated entity {entity_idx}: {list(fix.keys())}")
    
    scene_data["entities"] = entities
    return scene_data


def refine_scene_with_llm(original_prompt: str, scene_data: dict, feedback: dict) -> dict:
    """Ask LLM to regenerate scene with critic feedback."""
    print("  [Refinement] Asking LLM to improve scene...")
    
    refinement_prompt = f"""Original request: {original_prompt}

The current scene was analyzed and scored {feedback.get('score', '?')}/10.

Critic's analysis: {feedback.get('analysis', '')}

Issues found:
{json.dumps(feedback.get('issues', []), indent=2)}

Refinement instructions: {feedback.get('refinement_instructions', '')}

Please generate an IMPROVED scene that fixes these issues. Keep what's working well.
Return the complete scene JSON with all entities."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ENTITY_PROMPT},
            {"role": "user", "content": refinement_prompt}
        ],
        response_format={"type": "json_object"},
        timeout=30.0
    )
    
    return json.loads(response.choices[0].message.content)


def generate_scene_with_feedback(prompt: str, max_iterations: int = 3, 
                                  target_score: int = 8, auto_refine: bool = True) -> tuple:
    """
    Full pipeline with feedback loop.
    
    Returns: (final_meshes, final_scene_data, iteration_history)
    """
    history = []
    scene_data = None
    meshes = None
    
    for iteration in range(max_iterations):
        print(f"\n{'─'*50}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print('─'*50)
        
        # Stage 1: Generate or refine scene
        if iteration == 0:
            scene_data = get_entities_from_ai(prompt)
        elif history and history[-1].get("feedback", {}).get("should_refine"):
            scene_data = refine_scene_with_llm(prompt, scene_data, history[-1]["feedback"])
        else:
            break  # No refinement needed
        
        entity_count = len(scene_data.get("entities", []))
        print(f"  ✓ Stage 1: Got {entity_count} entities")
        
        # Stage 2: Decompose to primitives
        primitives = process_scene(scene_data)
        print(f"  ✓ Stage 2: Generated {len(primitives)} primitives")
        
        # Build meshes
        meshes = build_scene(primitives)
        print(f"  ✓ Built {len(meshes)} meshes")
        
        # Stage 3: Critic feedback (if auto_refine enabled)
        if auto_refine and iteration < max_iterations - 1:
            image_b64 = render_to_image(meshes)
            feedback = get_critic_feedback(image_b64, prompt, scene_data)
            
            score = feedback.get("score", 0)
            print(f"  ✓ Stage 3: Critic score = {score}/10")
            print(f"    Analysis: {feedback.get('analysis', '')[:100]}...")
            
            if feedback.get("issues"):
                print(f"    Issues found: {len(feedback['issues'])}")
                for issue in feedback["issues"][:3]:
                    print(f"      - [{issue.get('type')}] {issue.get('description', '')[:50]}")
            
            history.append({
                "iteration": iteration + 1,
                "entity_count": entity_count,
                "primitive_count": len(primitives),
                "mesh_count": len(meshes),
                "feedback": feedback
            })
            
            if score >= target_score:
                print(f"  ✓ Target score reached! Stopping refinement.")
                break
        else:
            history.append({
                "iteration": iteration + 1,
                "entity_count": entity_count,
                "primitive_count": len(primitives),
                "mesh_count": len(meshes),
                "feedback": None
            })
    
    return meshes, scene_data, history


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🏗️  AI 3D Scene Generator (Three-Stage Architecture)")
    print("=" * 60)
    print("Stage 1: LLM → Semantic Entities")
    print("Stage 2: Entity Agent → Primitive Objects (deterministic)")
    print("Stage 3: Vision Critic → Feedback Loop")
    print("-" * 60)
    print("Commands:")
    print("  [prompt]     - Generate scene (with auto-refinement)")
    print("  /norefine    - Disable auto-refinement for next prompt")
    print("  /refine      - Enable auto-refinement (default)")
    print("  /iterations N - Set max refinement iterations (default: 3)")
    print("  quit         - Exit")
    print("-" * 60)
    print("Examples:")
    print("  - 'a medieval village with a church and market square'")
    print("  - 'a park with trees, a pond, and benches'")
    print("  - 'a castle on a hill'")
    print()
    
    auto_refine = True
    max_iterations = 3
    
    while True:
        prompt = input("Describe a scene: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
        
        # Handle commands
        if prompt.startswith('/'):
            if prompt == '/norefine':
                auto_refine = False
                print("  Auto-refinement disabled for next prompt")
                continue
            elif prompt == '/refine':
                auto_refine = True
                print("  Auto-refinement enabled")
                continue
            elif prompt.startswith('/iterations'):
                try:
                    max_iterations = int(prompt.split()[1])
                    print(f"  Max iterations set to {max_iterations}")
                except:
                    print("  Usage: /iterations N")
                continue
            else:
                print("  Unknown command")
                continue
        
        print(f"\n🔨 Processing: {prompt}")
        try:
            meshes, scene_data, history = generate_scene_with_feedback(
                prompt, 
                max_iterations=max_iterations if auto_refine else 1,
                auto_refine=auto_refine
            )
            
            # Show final render
            print(f"\n  Opening 3D viewer...")
            plotter = pv.Plotter()
            plotter.background_color = "lightblue"
            
            for item in meshes:
                plotter.add_mesh(item["mesh"], color=item["color"], show_edges=False)
            
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.show()
            
            # Reset auto_refine if it was disabled
            auto_refine = True
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not prompt:
            continue
        
        print(f"\n🔨 Processing: {prompt}")
        try:
            # Stage 1: Get semantic entities
            scene_data = get_entities_from_ai(prompt)
            entity_count = len(scene_data.get("entities", []))
            print(f"  ✓ Stage 1: Got {entity_count} entities")
            if scene_data.get("scene_description"):
                print(f"    Description: {scene_data['scene_description']}")
            
            # Stage 2: Decompose to primitives
            primitives = process_scene(scene_data)
            print(f"  ✓ Stage 2: Generated {len(primitives)} primitives")
            
            # Render
            print("  Opening 3D viewer...\n")
            plotter = pv.Plotter()
            plotter.background_color = "lightblue"
            
            meshes = build_scene(primitives)
            for item in meshes:
                plotter.add_mesh(item["mesh"], color=item["color"], show_edges=False)
            
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.show()
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
